import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from labelpilot.exporting import export_project, preview_export
from labelpilot.export_jobs import ExportJobs
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


def ocr_shape(text="00.250 g", **attributes):
    return {"label": "text", "shape_type": "rectangle", "points": [[5, 8], [65, 28]],
            "text_raw": text, "readability": "readable", "value_role": "process_value",
            "value_decimal": "00.250", "unit": "g", "parent_id": "panel-001", **attributes}


def test_ocr_default_class_and_precise_text_survive_revision_and_reload(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    source = Path(project.item(item)["path"]).read_bytes()
    shape = ocr_shape("型号 ABC-001  00.250 g", custom_attribute={"keep": True})
    first = reviewed(store, project, item, [shape])
    assert project.names == ["text"] and first["document"]["shapes"] == [shape]
    restored = Store(store.home).detail(project.id, item)
    assert restored["document"]["shapes"] == [shape]
    assert restored["version"] == 1 and restored["status"] == "reviewed"
    assert Path(project.item(item)["path"]).read_bytes() == source
    with archive(store, project, "labelme") as exported:
        annotation = next(name for name in exported.namelist() if name.startswith("annotations/"))
        assert json.loads(exported.read(annotation))["shapes"] == [shape]


@pytest.mark.parametrize("attributes", [
    {"text_raw": ""}, {"text_raw": "  "}, {"text_raw": "###"}, {"text_raw": "123\t456"},
    {"text_raw": "123\n456"}, {"text_raw": "123\u2028abc"}, {"text_raw": 12}, {"readability": "blank"}, {"value_role": "guess"},
    {"value_decimal": 0.25}, {"value_decimal": "NaN"}, {"value_decimal": "0x250"},
    {"value_decimal": "1e3"}, {"readability": "partial"}, {"readability": "unreadable"},
    {"shape_type": "polygon", "points": [[5, 8], [65, 28], [65, 8], [5, 28]]},
    {"shape_type": "polygon", "points": [[5, 8], [65, 8], [5, 28]]},
])
def test_ocr_invalid_review_is_rejected_without_saving(tmp_path, attributes):
    store, project, item = project_with_image(tmp_path, "ocr")
    with pytest.raises(Problem) as error:
        reviewed(store, project, item, [ocr_shape(**attributes)])
    assert error.value.status == 422
    assert store.detail(project.id, item)["version"] == 0


def test_ocr_draft_can_wait_for_transcription_but_training_export_cannot(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    current = store.detail(project.id, item)
    document = {**current["document"], "shapes": [ocr_shape("", value_decimal=None)]}
    store.save(project.id, item, {**current, "document": document, "status": "draft", "expected_version": 0})
    assert store.detail(project.id, item)["document"]["shapes"][0]["text_raw"] == ""
    with pytest.raises(Problem, match="原文"):
        preview_export(store, project.id, {"format": "paddleocr", "reviewed_only": False})


def test_ocr_training_export_preserves_text_and_excludes_unreadable_recognition(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    readable = ocr_shape()
    partial = ocr_shape("1?3", readability="partial", value_decimal=None)
    unreadable = ocr_shape(None, readability="unreadable", value_decimal=None)
    reviewed(store, project, item, [readable, partial, unreadable])
    original = Path(project.item(item)["path"]).read_bytes()
    assert preview_export(store, project.id, {"format": "paddleocr"})["annotations"] == 3
    with archive(store, project, "paddleocr") as exported:
        image_name, encoded = exported.read("det/all.txt").decode().rstrip("\n").split("\t", 1)
        entries = json.loads(encoded)
        assert [entry["transcription"] for entry in entries] == ["00.250 g", "###", "###"]
        assert entries[0]["points"] == [[5, 8], [65, 8], [65, 28], [5, 28]]
        crop_name, text = exported.read("rec/all.txt").decode().rstrip("\n").split("\t", 1)
        assert text == "00.250 g" and exported.read(image_name) == original
        with Image.open(io.BytesIO(exported.read(crop_name))) as crop:
            assert crop.size == (60, 20) and crop.mode == "RGB"
        assert len([name for name in exported.namelist() if name.startswith("recognition/")]) == 1
        metadata = json.loads(exported.read(next(name for name in exported.namelist() if name.startswith("annotations/"))))
        assert metadata["shapes"] == [readable, partial, unreadable]
        manifest = json.loads(exported.read("export_manifest.json"))
        assert manifest["ocr"]["splits"]["all"] == {"images": 1, "recognition_crops": 1}
        assert manifest["integrity_check"]["passed"] is True
    assert Path(project.item(item)["path"]).read_bytes() == original


def test_ocr_confirmed_blank_is_not_zero_or_a_recognition_sample(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    with pytest.raises(Problem, match="空标注"):
        reviewed(store, project, item, [])
    reviewed(store, project, item, [], confirmed_empty=True)
    with archive(store, project, "paddleocr") as exported:
        assert json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1]) == []
        assert exported.read("rec/all.txt") == b""
        assert not any(name.startswith("recognition/") for name in exported.namelist())


@pytest.mark.parametrize("points", [
    [[5, 8], [65, 10], [60, 28], [8, 25]],
    [[8, 25], [60, 28], [65, 10], [5, 8]],
])
def test_ocr_slanted_quadrilateral_preserves_points_and_text(tmp_path, points):
    store, project, item = project_with_image(tmp_path, "ocr")
    shape = ocr_shape("-01.20 °C", shape_type="polygon", points=points, value_decimal="-01.20", unit="°C")
    reviewed(store, project, item, [shape])
    with archive(store, project, "paddleocr") as exported:
        entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
        assert entries == [{"transcription": "-01.20 °C", "points": points}]
        assert exported.read("rec/all.txt").decode().endswith("\t-01.20 °C\n")


def test_ocr_empty_text_is_not_auto_populated_with_zero(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    reviewed(store, project, item, [ocr_shape(None, readability="unreadable", value_decimal=None)])
    with archive(store, project, "paddleocr") as exported:
        assert exported.read("rec/all.txt") == b""
        entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
        assert entries[0]["transcription"] == "###"


def test_ocr_perspective_crop_is_not_mirrored_and_preserves_source_pixels():
    from PIL import ImageDraw
    from labelpilot.ocr import recognition_crop

    source = Image.new("RGB", (80, 60), "white")
    painter = ImageDraw.Draw(source)
    painter.rectangle((9, 9, 18, 18), fill="red")
    painter.rectangle((51, 9, 60, 18), fill="blue")
    before = source.tobytes()
    crop = recognition_crop(source, [[10, 10], [60, 10], [60, 30], [10, 30]])
    assert crop.size == (50, 20)
    assert crop.getpixel((2, 2)) == (255, 0, 0)
    assert crop.getpixel((47, 2)) == (0, 0, 255)
    assert source.tobytes() == before


def test_ocr_folder_export_inherits_original_split_for_all_crops(tmp_path):
    store, project, item = project_with_image(tmp_path, "ocr")
    for index in range(1, 4):
        Image.new("RGB", (80, 60), (index * 40, 0, 0)).save(project.root / f"frame{index}.png")
    store.rescan(project.id)
    project = store.project(project.id)
    for row in project.items:
        reviewed(store, project, row["id"], [ocr_shape()])
    destination = tmp_path / "destination"
    destination.mkdir()
    jobs = ExportJobs(store)
    record = jobs.start(project.id, {"format": "paddleocr", "destination": str(destination),
                                    "split": {"train": 50, "val": 25, "test": 25, "seed": 42}})
    jobs.worker.join(timeout=20)
    result = jobs.get(project.id, record["id"])
    assert result["status"] == "completed", result
    output = Path(result["result"]["directory"])
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    for subset, count in (("train", 2), ("val", 1), ("test", 1)):
        assert manifest["ocr"]["splits"][subset] == {"images": count, "recognition_crops": count}
        lines = (output / "rec" / f"{subset}.txt").read_text(encoding="utf-8").splitlines()
        for line in lines:
            name, text = line.split("\t", 1)
            assert name.startswith(f"recognition/{subset}/") and (output / name).is_file()
            assert text == "00.250 g"
    jobs.shutdown()


def test_ocr_rejects_yolo_formats_missing_images_augmentation_and_training(tmp_path):
    from labelpilot.inference import Inference, Jobs
    from labelpilot.training import TrainingJobs, training_selection

    store, project, item = project_with_image(tmp_path, "ocr")
    reviewed(store, project, item, [ocr_shape()])
    with pytest.raises(Problem, match="格式"):
        preview_export(store, project.id, {"format": "yolo"})
    with pytest.raises(Problem, match="包含原图"):
        preview_export(store, project.id, {"format": "paddleocr", "include_images": False})
    with pytest.raises(Problem):
        preview_export(store, project.id, {"format": "paddleocr", "augmentation": {"preset": "all", "copies": 1}})
    inference = Inference(store)
    jobs = Jobs(store, inference)
    with pytest.raises(Problem, match="OCR"):
        inference.predict(project.id, item, {})
    with pytest.raises(Problem, match="OCR"):
        jobs.start(project.id, {})
    with pytest.raises(Problem, match="OCR"):
        training_selection(store, project.id, {})
    with pytest.raises(Problem, match="OCR"):
        TrainingJobs(store, inference, jobs).start(project.id, {})


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


@pytest.mark.parametrize("task,format_name", [("segment", "yolo"), ("segment", "coco"),
                                               ("semantic", "mask"), ("hand_pose", "yolo"), ("hand_pose", "coco")])
def test_task_folder_export_preserves_annotations(tmp_path, task, format_name):
    store, project, item = project_with_image(tmp_path, task, ["part"] if task != "hand_pose" else None)
    shapes = [polygon("part", 4, 5, 35, 30)]
    if task == "hand_pose":
        shapes = [{"label": "hand", "shape_type": "hand", "points": [[10, 10], [60, 55]],
                   "keypoints": [[20, 20, 2], [30, 30, 1]] + [[0, 0, 0] for _ in range(HAND_COUNT - 2)]}]
    reviewed(store, project, item, shapes)
    jobs = ExportJobs(store)
    try:
        record = jobs.start(project.id, {"format": format_name, "destination": str(tmp_path)})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = jobs.get(project.id, record["id"])
            if record["status"] in ("completed", "failed"):
                break
            time.sleep(.01)
        assert record["status"] == "completed", record
        root = Path(record["result"]["directory"])
        manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["task_type"] == task and manifest["integrity_check"]["passed"]
        if task == "semantic":
            mask_path = next((root / "masks").glob("*.png"))
            with Image.open(mask_path) as mask:
                assert mask.size == (80, 60) and mask.getpixel((8, 8)) == 1
        elif format_name == "yolo":
            values = next((root / "labels").glob("*.txt")).read_text().split()
            assert len(values) == (5 + HAND_COUNT * 3 if task == "hand_pose" else 9)
        else:
            document = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
            assert len(document["annotations"]) == 1
    finally:
        jobs.shutdown()


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
