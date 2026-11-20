import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from labelpilot.exporting import export_project, preview_export
from labelpilot.store import Problem, Store, atomic_json
from labelpilot.tasks import HAND_COUNT, mask_shape


def project_with_image(tmp_path, task, classes=None):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (80, 60), "#213646").save(images / "frame.png")
    store = Store(tmp_path / "workspace")
    summary = store.register({"name": task, "task_type": task, "image_root": str(images), "classes": classes})
    project = store.project(summary["id"])
    return store, project, project.items[0]["id"]


def reviewed(store, project, item, shapes, *, confirmed_empty=False):
    current = store.detail(project.id, item)
    document = dict(current["document"], shapes=shapes)
    return store.save(project.id, item, {**current, "document": document, "status": "reviewed",
                                        "expected_version": current["version"], "confirmed_empty": confirmed_empty})


@pytest.mark.parametrize("labels", ["new_label\nbackground\n", "new_label\nexceeds_limit\n"])
def test_semantic_label_file_import_rejects_reserved_or_excess_classes_atomically(tmp_path, labels):
    store, project, _ = project_with_image(tmp_path, "semantic", [f"class_{index}" for index in range(253)])
    registry = store.registry_path.read_bytes()
    with pytest.raises(Problem, match="254"):
        store.import_labels(project.id, "classes.txt", labels)
    assert store.registry_path.read_bytes() == registry
    assert len(project.names) == 253
    assert store.import_labels(project.id, "classes.txt", "new_label\n")["added"] == 1
    assert project.names[-1] == "new_label" and len(project.names) == 254


def polygon(label, left, top, width, height):
    return {"label": label, "shape_type": "polygon", "points": [[left, top], [left + width, top],
            [left + width, top + height], [left, top + height]]}


def archive(store, project, format_name):
    result = export_project(store, project.id, {"format": format_name})
    return zipfile.ZipFile(store.home / "exports" / result["filename"])


def test_instance_polygons_export_as_distinct_same_class_objects(tmp_path):
    store, project, item = project_with_image(tmp_path, "segment", ["part"])
    first, second = polygon("part", 3, 4, 18, 21), polygon("part", 24, 20, 32, 29)
    with pytest.raises(Problem, match="不支持此标注形状"):
        reviewed(store, project, item, [{"label": "part", "shape_type": "rectangle", "points": [[2, 3], [8, 9]]}])
    reviewed(store, project, item, [first, second])
    assert preview_export(store, project.id, {"format": "yolo"})["blocking_issue"] is None
    with archive(store, project, "yolo") as exported:
        name = next(name for name in exported.namelist() if name.startswith("labels/"))
        rows = [line.split() for line in exported.read(name).decode().splitlines()]
        assert len(rows) == 2
        assert all(len(row) == 9 and row[0] == "0" for row in rows)
    with archive(store, project, "coco") as exported:
        entries = json.loads(exported.read("annotations.json"))["annotations"]
        assert [row["id"] for row in entries] == [1, 2]
        assert all(len(row["segmentation"][0]) == 8 for row in entries)


@pytest.mark.parametrize("format_name", ["yolo", "labelme", "coco"])
@pytest.mark.parametrize("renamed", [False, True])
def test_instance_rotation_and_label_alias_export_keep_every_polygon(tmp_path, format_name, renamed):
    store, project, item = project_with_image(tmp_path, "segment", ["part"])
    shapes = [polygon("part", 3, 4, 18, 21), polygon("part", 24, 20, 32, 29)]
    reviewed(store, project, item, shapes)
    if renamed:
        store.configs[0]["classes"] = ["part_1"]
        store.configs[0]["label_aliases"] = {"part": "part_1"}
        atomic_json(store.registry_path, store.configs)
        store = Store(store.home)
    with store.connect() as connection:
        before = {table: [tuple(row) for row in connection.execute("SELECT * FROM " + table)]
                  for table in ("annotations", "revisions")}
    options = {"format": format_name, "split": {"train": 100, "val": 0, "test": 0},
               "augmentation": {"preset": "rotation", "copies": 3}}
    assert preview_export(store, project.id, options)["blocking_issue"] is None
    result = export_project(store, project.id, options)
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as exported:
        manifest = json.loads(exported.read("export_manifest.json"))
        variants = [entry for entry in manifest["items"] if "augmentation" in entry]
        assert len(variants) == 3
        assert {entry["augmentation"]["operations"]["rotation_degrees"] for entry in variants} == {90, 180, 270}
        assert manifest["integrity_check"]["passed"]
        assert exported.read("classes.txt").decode().splitlines() == ["part_1" if renamed else "part"]
        for entry in variants:
            angle = entry["augmentation"]["operations"]["rotation_degrees"]
            width, height = (80, 60) if angle == 180 else (60, 80)
            transformed = [([[top, 80 - left] for left, top in shape["points"]] if angle == 90 else
                            [[80 - left, 60 - top] for left, top in shape["points"]] if angle == 180 else
                            [[60 - top, left] for left, top in shape["points"]]) for shape in shapes]
            with Image.open(io.BytesIO(exported.read(entry["image"]))) as image:
                assert image.size == (width, height)
            stem = Path(entry["image"]).stem
            if format_name == "yolo":
                rows = [line.split() for line in exported.read("labels/train/" + stem + ".txt").decode().splitlines()]
                assert len(rows) == 2
                for row, points in zip(rows, transformed):
                    assert row[0] == "0"
                    assert [float(value) for value in row[1:]] == pytest.approx(
                        [value / size for point in points for value, size in zip(point, (width, height))], abs=1e-10)
            elif format_name == "labelme":
                document = json.loads(exported.read("annotations/train/" + stem + ".json"))
                assert [shape["points"] for shape in document["shapes"]] == transformed
            else:
                document = json.loads(exported.read("annotations/train.json"))
                image_id = next(image["id"] for image in document["images"] if image["file_name"] == entry["image"])
                annotations = [shape for shape in document["annotations"] if shape["image_id"] == image_id]
                assert len(annotations) == 2
                assert [shape["segmentation"][0] for shape in annotations] == [
                    [value for point in points for value in point] for points in transformed]
    with store.connect() as connection:
        assert before == {table: [tuple(row) for row in connection.execute("SELECT * FROM " + table)]
                          for table in ("annotations", "revisions")}


def test_semantic_mask_has_class_ids_ordered_paint_and_eraser(tmp_path):
    store, project, item = project_with_image(tmp_path, "semantic", ["road", "curb"])
    mask = Image.new("L", (80, 60))
    mask.putpixel((72, 8), 255)
    reviewed(store, project, item, [
        polygon("road", 4, 4, 38, 38),
        {"label": "curb", "shape_type": "brush", "points": [[26, 26]], "radius": 7},
        {"label": "road", "shape_type": "brush", "points": [[26, 26]], "radius": 2, "erase": True},
        {**mask_shape(mask), "label": "road"},
    ])
    with archive(store, project, "mask") as exported:
        manifest = json.loads(exported.read("export_manifest.json"))
        assert manifest["mask_classes"] == ["background", "road", "curb"]
        stem = Path(manifest["items"][0]["image"]).stem
        with Image.open(io.BytesIO(exported.read(f"masks/{stem}.png"))) as output:
            assert output.mode == "L" and output.size == (80, 60)
            assert [output.getpixel(point) for point in [(0, 0), (8, 8), (26, 31), (26, 26), (72, 8)]] == [0, 1, 2, 0, 1]

    erased = [polygon("road", 12, 12, 5, 5),
              {"label": "road", "shape_type": "brush", "points": [[15, 15]], "radius": 14, "erase": True}]
    with pytest.raises(Problem, match="空标注"):
        reviewed(store, project, item, erased)
    assert reviewed(store, project, item, erased, confirmed_empty=True)["document"]["labelpilot"]["negative_sample_confirmed"]
    assert preview_export(store, project.id, {"format": "mask"})["confirmed_empty"] == 1


def test_hand_visibility_and_schema_survive_yolo_and_coco(tmp_path):
    store, project, item = project_with_image(tmp_path, "hand_pose")
    assert project.names == ["hand"]
    assert len(project.summary()["keypoints"]) == HAND_COUNT == 21
    points = [[30, 45, 2], [34, 30, 1]] + [[0, 0, 0] for _ in range(19)]
    hand = {"label": "hand", "shape_type": "hand", "points": [[10, 10], [60, 55]], "keypoints": points}
    with pytest.raises(Problem, match="需要 21 个"):
        reviewed(store, project, item, [{**hand, "keypoints": points[:-1]}])
    with pytest.raises(Problem, match="位于手框内"):
        reviewed(store, project, item, [{**hand, "points": [[10, 10], [25, 40]]}])
    reviewed(store, project, item, [hand])
    with archive(store, project, "yolo") as exported:
        assert b"kpt_shape: [21, 3]" in exported.read("dataset.yaml")
        name = next(name for name in exported.namelist() if name.startswith("labels/"))
        row = list(map(float, exported.read(name).split()))
        assert len(row) == 5 + 3 * HAND_COUNT
        assert row[7::3] == [2, 1] + [0] * 19
    with archive(store, project, "coco") as exported:
        data = json.loads(exported.read("annotations.json"))
        assert data["annotations"][0]["num_keypoints"] == 2
        assert len(data["categories"][0]["keypoints"]) == HAND_COUNT
        assert data["categories"][0]["skeleton"][0] == [1, 2]


def test_atomic_json_retries_windows_sharing_violation(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    replace = Path.replace
    calls = []

    def transient_sharing_lock(path, target):
        calls.append(target)
        if len(calls) == 1:
            error = PermissionError("file in use")
            error.winerror = 32
            raise error
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", transient_sharing_lock)
    monkeypatch.setattr("labelpilot.store.time.sleep", lambda seconds: None)
    atomic_json(destination, {"epoch": 1})
    assert calls == [destination, destination]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"epoch": 1}
