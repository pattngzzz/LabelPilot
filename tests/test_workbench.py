import collections
import copy
import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot.augmentation import PRESETS, augmented_variants, plan_augmentation, validate_augmented_sample
from labelpilot.exporting import export_project, plan_splits, preview_export
from labelpilot import model_files, native_picker
from labelpilot import server
from labelpilot.inference import Inference, Jobs, model_class_names
from labelpilot.server import create_app
from labelpilot.store import Problem, Project, Store, digest


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    native = tmp_path / "native"
    predictions = tmp_path / "predictions"
    home = tmp_path / "tool"
    for directory in (images, native, predictions, home):
        directory.mkdir()
    for name in ("human", "model", "empty", "another"):
        Image.new("RGB", (240, 120), "#777777").save(images / f"{name}.jpg")
    document = {"version": "4.0.0-beta.5", "imageWidth": 240, "imageHeight": 120,
                "imagePath": "human.jpg", "checked": True, "flags": {"original": True},
                "custom_top_level": {"preserve": 1}, "imageData": None,
                "shapes": [{"label": "object", "shape_type": "rectangle", "points": [[12.5, 10], [110, 70]],
                            "description": "readability=unreadable", "attributes": {"occlusion": "partial"},
                            "group_id": 9, "score": None}]}
    (native / "human.json").write_text(json.dumps(document), encoding="utf-8")
    model_document = copy.deepcopy(document)
    model_document["preannotation"] = {"human_reviewed": False}
    (predictions / "model.json").write_text(json.dumps(model_document), encoding="utf-8")
    config = {"id": "test", "name": "Test dataset", "image_root": str(images),
              "annotation_root": str(native), "preannotation_root": str(predictions), "classes": ["object"]}
    (home / "projects.json").write_text(json.dumps([config]), encoding="utf-8")
    return home, images, native, predictions


@pytest.fixture
def store(dataset):
    return Store(dataset[0])


def item(store, name):
    return next(row for row in store.project("test").items if row["photo_id"] == name)


def payload(store, name, status="draft"):
    current = store.detail("test", item(store, name)["id"])
    return {**current, "expected_version": current["version"], "status": status}


@pytest.fixture
def reviewed_store(store):
    for row in store.project("test").items:
        data = payload(store, row["photo_id"], "reviewed")
        data["confirmed_empty"] = True
        store.save("test", row["id"], data)
    return store


@pytest.mark.parametrize("format_name", ["yolo", "labelme", "coco"])
@pytest.mark.parametrize("include_images", [True, False])
def test_split_export_layout_and_preservation(reviewed_store, dataset, format_name, include_images):
    store = reviewed_store
    sources = {path: digest(path) for folder in dataset[1:] for path in folder.rglob("*") if path.is_file()}
    with store.connect() as connection:
        annotations = [tuple(row) for row in connection.execute("SELECT * FROM annotations ORDER BY item")]
        revisions = [tuple(row) for row in connection.execute("SELECT * FROM revisions ORDER BY item, version")]
    options = {"format": format_name, "include_images": include_images,
               "split": {"train": 50, "val": 25, "test": 25, "seed": 42}}
    result = export_project(store, "test", options)
    assert result["split_counts"] == {"train": 2, "val": 1, "test": 1}
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("export_manifest.json"))
        assert len(manifest["items"]) == len({entry["id"] for entry in manifest["items"]}) == 4
        assert manifest["split_config"] == {**options["split"], "unit": "image", "method": "seeded_shuffle"}
        assert dict(collections.Counter(entry["split"] for entry in manifest["items"])) == result["split_counts"]
        assert manifest["source_writes"] == 0 and manifest["include_images"] == include_images
        assert archive.read("classes.txt") == b"object\n"
        for entry in manifest["items"]:
            assert entry["status"] == "reviewed"
            image_path = PurePosixPath(entry["image"])
            assert image_path.parts[:2] == ("images", entry["split"])
            assert (entry["image"] in names) == include_images
            if include_images:
                assert archive.read(entry["image"]) == Path(store.project("test").item(entry["id"])["path"]).read_bytes()
            if format_name == "yolo":
                label = image_path.with_suffix(".txt").as_posix().replace("images/", "labels/", 1)
                assert label in names
                if entry["photo_id"] in ("empty", "another"):
                    assert archive.read(label) == b""
            elif format_name == "labelme":
                annotation = image_path.with_suffix(".json").as_posix().replace("images/", "annotations/", 1)
                document = json.loads(archive.read(annotation))
                assert document["imagePath"] == "../../" + entry["image"]
                assert document["imageData"] is None
        if format_name == "yolo":
            yaml = archive.read("dataset.yaml").decode()
            assert all(f"{name}: images/{name}\n" in yaml for name in ("train", "val", "test"))
        elif format_name == "coco":
            for subset, count in result["split_counts"].items():
                document = json.loads(archive.read(f"annotations/{subset}.json"))
                assert len(document["images"]) == count
                assert all(image["file_name"].startswith(f"images/{subset}/") for image in document["images"])
                assert all(annotation["image_id"] in {image["id"] for image in document["images"]}
                           for annotation in document["annotations"])
                assert document["categories"] == [{"id": 1, "name": "object"}]
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())
    with store.connect() as connection:
        assert annotations == [tuple(row) for row in connection.execute("SELECT * FROM annotations ORDER BY item")]
        assert revisions == [tuple(row) for row in connection.execute("SELECT * FROM revisions ORDER BY item, version")]


def test_split_is_reproducible_and_seeded():
    rows = [{"id": str(index)} for index in range(100)]
    settings = {"split": {"train": 80, "val": 10, "test": 10, "seed": 42}}
    assignments, config, counts = plan_splits(rows, settings)
    assert plan_splits(list(reversed(rows)), settings) == (assignments, config, counts)
    assert counts == {"train": 80, "val": 10, "test": 10}
    assert assignments != plan_splits(rows, {"split": {**settings["split"], "seed": 43}})[0]


@pytest.mark.parametrize("unit", ["source_group", "directory"])
def test_grouped_split_keeps_groups_whole_and_is_reproducible(unit):
    groups = {"day1/camera": 6, "day2/camera": 3, "day3/camera": 1}
    rows = [{"id": f"{group}/{index}", "source_group": group, "relative": f"{group}/{index}.jpg"}
            for group, size in groups.items() for index in range(size)]
    options = {"split": {"train": 60, "val": 30, "test": 10, "seed": 42, "unit": unit}}
    assignments, config, counts = plan_splits(rows, options)
    assert plan_splits(list(reversed(rows)), options) == (assignments, config, counts)
    assert config["unit"] == unit
    assert counts == {"train": 6, "val": 3, "test": 1}
    assert all(len({assignments[row["id"]] for row in rows if row["source_group"] == group}) == 1 for group in groups)


def test_grouped_split_requires_known_and_sufficient_groups():
    rows = [{"id": str(index), "source_group": "video-a"} for index in range(5)]
    options = {"split": {"train": 80, "val": 20, "test": 0, "unit": "source_group"}}
    with pytest.raises(Problem, match="只有 1 个可用分组"):
        plan_splits(rows, options)
    rows[0]["source_group"] = " unassigned "
    with pytest.raises(Problem, match="1 张图片没有来源组"):
        plan_splits(rows, options)
    rows[0]["source_group"] = "video-b"
    _, _, counts = plan_splits(rows, options)
    assert counts == {"train": 4, "val": 1, "test": 0}

@pytest.mark.parametrize("preset,copies", [(preset, 2) for preset in PRESETS] + [("all", 1), ("all", 3)])
@pytest.mark.parametrize("format_name", ["yolo", "labelme", "coco"])
def test_augmentation_export_train_only_and_preservation(reviewed_store, dataset, preset, copies, format_name):
    store = reviewed_store
    sources = {path: digest(path) for folder in dataset[1:] for path in folder.rglob("*") if path.is_file()}
    with store.connect() as connection:
        annotations = [tuple(row) for row in connection.execute("SELECT * FROM annotations ORDER BY item")]
        revisions = [tuple(row) for row in connection.execute("SELECT * FROM revisions ORDER BY item, version")]
    options = {"format": format_name, "split": {"train": 50, "val": 25, "test": 25, "seed": 42},
               "augmentation": {"preset": preset, "copies": copies, "seed": 43}}
    result = export_project(store, "test", options)
    presets = list(PRESETS) if preset == "all" else [preset]
    augmented_count = 2 * copies * len(presets)
    total = 4 + augmented_count
    assert (result["images"], result["original_images"], result["augmented_images"]) == (total, 4, augmented_count)
    assert result["source_split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert result["split_counts"] == {"train": 2 + augmented_count, "val": 1, "test": 1}
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        assert manifest["augmentation"]["scope"] == "train_only"
        assert manifest["augmentation"]["presets"] == presets
        assert manifest["augmentation"]["variants_per_image"] == copies * len(presets)
        assert result["integrity_check"] == manifest["integrity_check"]
        assert manifest["integrity_check"]["passed"] and manifest["integrity_check"]["images_checked"] == total
        assert manifest["integrity_check"]["annotations_checked"] == sum(entry["annotation_count"] for entry in manifest["items"])
        assert len(manifest["items"]) == len({entry["image"] for entry in manifest["items"]}) == total
        assert len({entry["id"] for entry in manifest["items"]}) == total
        originals = {entry["id"]: entry for entry in manifest["items"] if "augmentation" not in entry}
        variants = [entry for entry in manifest["items"] if "augmentation" in entry]
        assert len(variants) == augmented_count
        for source in originals.values():
            actual = {(entry["augmentation"]["preset"], entry["augmentation"]["variant"])
                      for entry in variants if entry["source_id"] == source["id"]}
            expected = {(name, index) for name in presets for index in range(1, copies + 1)} if source["split"] == "train" else set()
            assert actual == expected
        for entry in originals.values():
            assert archive.read(entry["image"]) == Path(store.project("test").item(entry["id"])["path"]).read_bytes()
        for entry in variants:
            source = originals[entry["source_id"]]
            assert entry["split"] == source["split"] == "train"
            assert not entry["human_reviewed"] and entry["status"] == "augmented"
            contents = archive.read(entry["image"])
            angle = entry["augmentation"]["operations"].get("rotation_degrees", 0)
            width, height = (120, 240) if angle in (90, 270) else (240, 120)
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"]
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
                assert image.size == (width, height) and image.format == "JPEG"
            original = store.detail("test", entry["source_id"])["document"]
            expected = copy.deepcopy(original["shapes"])
            if entry["augmentation"]["preset"] == "horizontal":
                for shape in expected:
                    shape["points"] = [[240 - point[0], point[1]] for point in shape["points"]]
            elif angle:
                for shape in expected:
                    if angle == 90:
                        shape["points"] = [[point[1], 240 - point[0]] for point in shape["points"]]
                    elif angle == 180:
                        shape["points"] = [[240 - point[0], 120 - point[1]] for point in shape["points"]]
                    else:
                        shape["points"] = [[120 - point[1], point[0]] for point in shape["points"]]
            stem = Path(entry["image"]).stem
            if format_name == "labelme":
                document = json.loads(archive.read(f"annotations/train/{stem}.json"))
                assert document["shapes"] == expected
                assert (document["imageWidth"], document["imageHeight"]) == (width, height)
                assert document["flags"] == original["flags"]
                if "custom_top_level" in original:
                    assert document["custom_top_level"] == original["custom_top_level"]
                assert document["imagePath"] == "../../" + entry["image"] and document["imageData"] is None
                assert not document["checked"] and not document["labelpilot"]["human_reviewed"]
                assert document["labelpilot"]["image_sha256"] == entry["sha256"]
                assert document["labelpilot"]["augmentation"] == entry["augmentation"]
            elif format_name == "yolo":
                lines = archive.read(f"labels/train/{stem}.txt").decode().splitlines()
                assert len(lines) == len(expected)
                for line, shape in zip(lines, expected):
                    values = line.split()
                    xs, ys = zip(*shape["points"])
                    assert values[0] == "0"
                    assert list(map(float, values[1:])) == pytest.approx([
                        (min(xs) + max(xs)) / (2 * width), (min(ys) + max(ys)) / (2 * height),
                        (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height])
            else:
                document = json.loads(archive.read("annotations/train.json"))
                image_id = next(image["id"] for image in document["images"] if image["file_name"] == entry["image"])
                metadata = next(image for image in document["images"] if image["id"] == image_id)
                assert (metadata["width"], metadata["height"]) == (width, height)
                shapes = [shape for shape in document["annotations"] if shape["image_id"] == image_id]
                assert len(shapes) == len(expected)
                for annotation, shape in zip(shapes, expected):
                    xs, ys = zip(*shape["points"])
                    assert annotation["bbox"] == pytest.approx([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
        if format_name == "coco":
            documents = [json.loads(archive.read(f"annotations/{subset}.json")) for subset in ("train", "val", "test")]
            ids = [image["id"] for document in documents for image in document["images"]]
            assert len(ids) == len(set(ids)) == total
            assert [len(document["images"]) for document in documents] == [2 + augmented_count, 1, 1]
        repeated = export_project(store, "test", options)
        with zipfile.ZipFile(store.home / "exports" / repeated["filename"]) as other:
            assert all(archive.read(entry["image"]) == other.read(entry["image"]) for entry in variants)
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())
    with store.connect() as connection:
        assert annotations == [tuple(row) for row in connection.execute("SELECT * FROM annotations ORDER BY item")]
        assert revisions == [tuple(row) for row in connection.execute("SELECT * FROM revisions ORDER BY item, version")]


def test_all_augmentation_matches_independent_presets_and_seeded_split(reviewed_store):
    options = {"format": "labelme", "split": {"train": 50, "val": 25, "test": 25, "seed": 42},
               "augmentation": {"preset": "all", "copies": 2, "seed": 43}}
    result = export_project(reviewed_store, "test", options)
    with zipfile.ZipFile(reviewed_store.home / "exports" / result["filename"]) as combined:
        manifest = json.loads(combined.read("export_manifest.json"))
        variants = {entry["id"]: entry for entry in manifest["items"] if "augmentation" in entry}
        for preset in PRESETS:
            single = export_project(reviewed_store, "test", {**options, "augmentation": {**options["augmentation"], "preset": preset}})
            with zipfile.ZipFile(reviewed_store.home / "exports" / single["filename"]) as archive:
                single_manifest = json.loads(archive.read("export_manifest.json"))
                for entry in single_manifest["items"]:
                    if "augmentation" in entry:
                        assert entry["augmentation"] == variants[entry["id"]]["augmentation"]
                        assert combined.read(entry["image"]) == archive.read(entry["image"])
                        annotation = f"annotations/train/{Path(entry['image']).stem}.json"
                        assert combined.read(annotation) == archive.read(annotation)
        changed = export_project(reviewed_store, "test", {**options, "augmentation": {**options["augmentation"], "seed": 44}})
        with zipfile.ZipFile(reviewed_store.home / "exports" / changed["filename"]) as archive:
            changed_manifest = json.loads(archive.read("export_manifest.json"))
            assert {entry["id"]: entry["split"] for entry in changed_manifest["items"]} == {
                entry["id"]: entry["split"] for entry in manifest["items"]}
            for entry in changed_manifest["items"]:
                if "augmentation" in entry and entry["augmentation"]["preset"] != "rotation":
                    assert entry["augmentation"]["operations"] != variants[entry["id"]]["augmentation"]["operations"]


@pytest.mark.parametrize("format_name", ["labelme", "coco", "yolo"])
def test_augmentation_flip_geometry_and_pixels(store, format_name):
    source = Path(store.project("test").item(item(store, "human")["id"])["path"])
    with Image.new("RGB", (240, 120), "red") as image:
        image.paste("blue", (120, 0, 240, 120))
        image.save(source)
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"].append({"label": "object", "shape_type": "rectangle",
                                     "points": [[0, 0], [240, 0], [240, 120], [0, 120]]})
    if format_name != "yolo":
        data["document"]["shapes"].append({"label": "object", "shape_type": "polygon",
                                         "points": [[10, 10], [80, 10], [80, 50]]})
    store.save("test", item(store, "human")["id"], data)
    options = {"format": format_name, "split": {"train": 100, "val": 0, "test": 0},
               "augmentation": {"preset": "horizontal"}}
    result = export_project(store, "test", options)
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        variant = next(entry for entry in manifest["items"] if "augmentation" in entry)
        with Image.open(io.BytesIO(archive.read(variant["image"]))) as image:
            assert image.getpixel((20, 60))[2] > image.getpixel((20, 60))[0] + 100
            assert image.getpixel((220, 60))[0] > image.getpixel((220, 60))[2] + 100
        if format_name == "labelme":
            document = json.loads(archive.read("annotations/train/" + Path(variant["image"]).stem + ".json"))
            for actual, original in zip(document["shapes"], data["document"]["shapes"]):
                expected = [[240 - point[0], point[1]] for point in original["points"]]
                if original["shape_type"] == "rectangle" and len(expected) == 4:
                    xs, ys = zip(*expected)
                    expected = [[min(xs), min(ys)], [max(xs), max(ys)]]
                assert actual["points"] == expected
                assert actual["label"] == original["label"]
        elif format_name == "coco":
            document = json.loads(archive.read("annotations/train.json"))
            polygon = document["annotations"][-1]
            assert polygon["segmentation"] == [[230, 10, 160, 10, 160, 50]]
            assert polygon["bbox"] == [160, 10, 70, 40] and polygon["area"] == 1400
        else:
            lines = archive.read("labels/train/" + Path(variant["image"]).stem + ".txt").decode().splitlines()
            assert list(map(float, lines[-1].split()[1:])) == [0.5, 0.5, 1.0, 1.0]


@pytest.mark.parametrize("format_name", ["labelme", "coco", "yolo"])
def test_rotation_all_angles_geometry_pixels_and_export_dimensions(store, format_name):
    source = Path(store.project("test").item(item(store, "human")["id"])["path"])
    with Image.new("RGB", (240, 120), "red") as image:
        image.paste("#00ff00", (120, 0, 240, 60))
        image.paste("blue", (0, 60, 120, 120))
        image.paste("yellow", (120, 60, 240, 120))
        image.save(source)
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"].append({"label": "object", "shape_type": "rectangle",
                                         "points": [[0, 0], [240, 0], [240, 120], [0, 120]]})
    if format_name != "yolo":
        data["document"]["shapes"].append({"label": "object", "shape_type": "polygon",
                                             "points": [[10, 10], [80, 10], [80, 50]]})
    store.save("test", item(store, "human")["id"], data)
    saved = store.detail("test", item(store, "human")["id"])
    fingerprint = digest(source)
    expected_points = {90: [[10, 227.5], [70, 130]], 180: [[227.5, 110], [130, 50]],
                       270: [[110, 12.5], [50, 110]]}
    expected_polygons = {90: [[10, 230], [10, 160], [50, 160]],
                         180: [[230, 110], [160, 110], [160, 70]],
                         270: [[110, 10], [110, 80], [70, 80]]}
    expected_colors = {90: (0, 255, 0), 180: (255, 255, 0), 270: (0, 0, 255)}
    result = export_project(store, "test", {"format": format_name, "split": {"train": 100, "val": 0, "test": 0},
                                            "augmentation": {"preset": "rotation", "copies": 3}})
    assert result["images"] == 4 and result["integrity_check"]["augmented_images_checked"] == 3
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        variants = [entry for entry in json.loads(archive.read("export_manifest.json"))["items"] if "augmentation" in entry]
        assert {entry["augmentation"]["operations"]["rotation_degrees"] for entry in variants} == {90, 180, 270}
        for entry in variants:
            angle = entry["augmentation"]["operations"]["rotation_degrees"]
            width, height = (120, 240) if angle != 180 else (240, 120)
            with Image.open(io.BytesIO(archive.read(entry["image"]))) as image:
                assert image.size == (width, height)
                assert max(abs(actual - expected) for actual, expected in zip(image.getpixel((20, 20)), expected_colors[angle])) < 15
            xs, ys = zip(*expected_points[angle])
            bounds = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if format_name == "labelme":
                document = json.loads(archive.read("annotations/train/" + Path(entry["image"]).stem + ".json"))
                assert (document["imageWidth"], document["imageHeight"]) == (width, height)
                assert document["shapes"][0] == {**data["document"]["shapes"][0], "points": expected_points[angle]}
                assert document["shapes"][-1]["points"] == expected_polygons[angle]
                border = document["shapes"][1]["points"]
                assert border == [[0, 0], [width, height]]
            elif format_name == "coco":
                document = json.loads(archive.read("annotations/train.json"))
                image = next(image for image in document["images"] if image["file_name"] == entry["image"])
                assert (image["width"], image["height"]) == (width, height)
                annotations = [shape for shape in document["annotations"] if shape["image_id"] == image["id"]]
                assert annotations[0]["bbox"] == bounds
                assert annotations[1]["bbox"] == [0, 0, width, height]
                assert annotations[-1]["segmentation"] == [[coordinate for point in expected_polygons[angle] for coordinate in point]]
                assert annotations[-1]["area"] == 1400
            else:
                lines = archive.read("labels/train/" + Path(entry["image"]).stem + ".txt").decode().splitlines()
                assert len(lines) == 2
                assert list(map(float, lines[0].split()[1:])) == pytest.approx([
                    (min(xs) + max(xs)) / (2 * width), (min(ys) + max(ys)) / (2 * height),
                    bounds[2] / width, bounds[3] / height])
                assert list(map(float, lines[-1].split()[1:])) == [0.5, 0.5, 1.0, 1.0]
    assert digest(source) == fingerprint
    assert store.detail("test", item(store, "human")["id"]) == saved


@pytest.mark.parametrize("corruption", ["angle", "missing_angle", "variant", "size", "unrotated", "missing", "shift"])
def test_rotation_integrity_rejects_inconsistent_geometry(store, corruption):
    current = store.detail("test", item(store, "human")["id"])
    source = current["document"]
    settings = plan_augmentation({"split": {}, "augmentation": {"preset": "rotation", "copies": 3}})
    path = store.project("test").item(current["item"]["id"])["path"]
    variants = augmented_variants(path, source, "human", current["image_sha256"], settings)
    _, contents, augmented, provenance = next(entry for entry in variants if entry[3]["operations"]["rotation_degrees"] == 90)
    if corruption == "angle":
        provenance["operations"]["rotation_degrees"] = 180
    elif corruption == "missing_angle":
        provenance["operations"].pop("rotation_degrees")
    elif corruption == "variant":
        provenance["variant"] = True
    elif corruption == "size":
        augmented["imageWidth"], augmented["imageHeight"] = 240, 120
    elif corruption == "unrotated":
        augmented["shapes"] = copy.deepcopy(source["shapes"])
    elif corruption == "missing":
        augmented["shapes"].clear()
    else:
        augmented["shapes"][0]["points"][0][0] += 1
    with pytest.raises(Problem, match="已中止导出"):
        validate_augmented_sample(source, augmented, contents, settings, provenance)


@pytest.mark.parametrize("settings", [False, [], {}, {"preset": "unknown"}, {"preset": []},
    *({"preset": "gentle", "copies": value} for value in (False, 0, 4, 1.5, "1")),
    *({"preset": "gentle", "seed": value} for value in (True, -1, 4294967296, 1.5, "42")),
    {"preset": "gentle", "unknown": 1}])
def test_invalid_augmentation_before_export(reviewed_store, settings):
    with pytest.raises(Problem) as error:
        export_project(reviewed_store, "test", {"split": {"train": 100, "val": 0, "test": 0}, "augmentation": settings})
    assert error.value.status == 422
    assert not (reviewed_store.home / "exports").exists()


@pytest.mark.parametrize("extra", [{"split": None}, {"include_images": False}, {"reviewed_only": False}])
def test_augmentation_requires_split_images_and_review(reviewed_store, extra):
    options = {"split": {"train": 100, "val": 0, "test": 0}, "augmentation": {"preset": "gentle"}, **extra}
    with pytest.raises(Problem) as error:
        export_project(reviewed_store, "test", options)
    assert error.value.status == 422
    assert not (reviewed_store.home / "exports").exists()


@pytest.mark.parametrize("preset", ["gentle", "rotation"])
def test_augmentation_seed_and_copy_prefix_stability(store, preset):
    current = store.detail("test", item(store, "human")["id"])
    source = store.project("test").item(current["item"]["id"])["path"]
    options = {"split": {"train": 100, "val": 0, "test": 0}, "augmentation": {"preset": preset, "copies": 3}}
    settings = plan_augmentation(options)
    before = copy.deepcopy(current["document"])
    variants = list(augmented_variants(source, before, "human", current["image_sha256"], settings))
    fewer = list(augmented_variants(source, before, "human", current["image_sha256"], {**settings, "copies": 1}))
    changed = list(augmented_variants(source, before, "human", current["image_sha256"], {**settings, "seed": 43}))
    assert before == current["document"] and variants[0] == fewer[0]
    if preset == "rotation":
        assert {entry[3]["operations"]["rotation_degrees"] for entry in variants} == {90, 180, 270}
        sequences = {tuple(entry[3]["operations"]["rotation_degrees"] for entry in
                           augmented_variants(source, before, "human", current["image_sha256"], {**settings, "seed": seed}))
                     for seed in range(8)}
        assert len(sequences) > 1
    else:
        assert [entry[3]["operations"] for entry in variants] != [entry[3]["operations"] for entry in changed]
        assert len({entry[1] for entry in variants}) == 3


@pytest.mark.parametrize("preset,multiplier", [("gentle", 1500), ("all", 400)])
def test_augmentation_limit_and_failed_decode_cleanup(reviewed_store, monkeypatch, preset, multiplier):
    options = {"split": {"train": 100, "val": 0, "test": 0}, "augmentation": {"preset": preset, "copies": 3}}
    listing = reviewed_store.listing
    def large_listing(*args, **kwargs):
        rows = listing(*args, **kwargs)["items"] * multiplier
        return {"items": rows, "filtered": len(rows)}

    monkeypatch.setattr(reviewed_store, "listing", large_listing)
    with pytest.raises(Problem, match="20000"):
        export_project(reviewed_store, "test", options)
    assert not (reviewed_store.home / "exports").exists()
    monkeypatch.setattr(reviewed_store, "listing", listing)
    def fail_decode(*args, **kwargs):
        raise OSError("simulated augmentation decode failure")
    monkeypatch.setattr("labelpilot.exporting.augmented_variants", fail_decode)
    with pytest.raises(OSError, match="simulated"):
        export_project(reviewed_store, "test", options)
    assert not list((reviewed_store.home / "exports").glob("*.zip"))


@pytest.mark.parametrize("corruption", ["shift", "missing", "label"])
@pytest.mark.parametrize("preset", ["horizontal", "rotation", "all"])
def test_augmented_box_shift_missing_or_label_change_aborts_export(store, monkeypatch, corruption, preset):
    import labelpilot.exporting as exporting

    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    source_generator = exporting.augmented_variants
    def corrupted_variants(*args, **kwargs):
        for index, contents, augmented, provenance in source_generator(*args, **kwargs):
            if augmented["shapes"] and provenance["preset"] == ("rotation" if preset == "rotation" else "horizontal"):
                if corruption == "shift":
                    augmented["shapes"][0]["points"][0][0] += 1
                elif corruption == "missing":
                    augmented["shapes"].pop()
                else:
                    augmented["shapes"][0]["label"] = "missing-class"
            yield index, contents, augmented, provenance

    monkeypatch.setattr(exporting, "augmented_variants", corrupted_variants)
    with pytest.raises(Problem, match="对象数量|标签、坐标"):
        export_project(store, "test", {"split": {"train": 100, "val": 0, "test": 0},
                                        "augmentation": {"preset": preset}})
    assert not list((store.home / "exports").glob("*.zip"))


@pytest.mark.parametrize("format_name", ["labelme", "coco", "yolo"])
def test_augmented_serialized_annotation_corruption_aborts_export(store, monkeypatch, format_name):
    import labelpilot.exporting as exporting

    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    original_write = exporting.write_annotation
    def corrupted_write(archive, selected_format, *args, **kwargs):
        original_write(archive, selected_format, *args, **kwargs)
        image_name = args[2]
        subset = args[3]
        split_config = args[4]
        if "__aug_" not in image_name:
            return
        suffix = f"/{subset}" if split_config else ""
        base = Path(image_name).stem
        if selected_format == "labelme":
            archive.writestr(f"annotations{suffix}/{base}.json", "{}")
        elif selected_format == "yolo":
            archive.writestr(f"labels{suffix}/{base}.txt", "")
        else:
            document = args[5][subset]
            if document["annotations"]:
                document["annotations"].pop()

    monkeypatch.setattr(exporting, "write_annotation", corrupted_write)
    with pytest.raises(Problem, match="回读校验失败"):
        export_project(store, "test", {"format": format_name, "split": {"train": 100, "val": 0, "test": 0},
                                        "augmentation": {"preset": "horizontal"}})
    assert not list((store.home / "exports").glob("*.zip"))


@pytest.mark.parametrize("corruption", ["dimensions", "undecodable", "checksum", "extra", "attributes", "operation"])
def test_augmented_sample_integrity_rejects_corruption(store, corruption):
    current = store.detail("test", item(store, "human")["id"])
    source = current["document"]
    settings = plan_augmentation({"split": {}, "augmentation": {"preset": "horizontal"}})
    path = store.project("test").item(current["item"]["id"])["path"]
    _, contents, augmented, provenance = next(augmented_variants(path, source, "human", current["image_sha256"], settings))
    if corruption == "dimensions":
        with io.BytesIO() as buffer:
            Image.new("RGB", (120, 60)).save(buffer, format="JPEG")
            contents = buffer.getvalue()
        augmented["labelpilot"]["image_sha256"] = hashlib.sha256(contents).hexdigest()
    elif corruption == "undecodable":
        contents = b"invalid image"
    elif corruption == "checksum":
        augmented["labelpilot"]["image_sha256"] = "incorrect"
    elif corruption == "extra":
        augmented["shapes"].append(copy.deepcopy(augmented["shapes"][0]))
    elif corruption == "attributes":
        augmented["shapes"][0]["attributes"]["occlusion"] = "incorrect"
    else:
        provenance["operations"]["horizontal_flip"] = False
    with pytest.raises(Problem, match="已中止导出"):
        validate_augmented_sample(source, augmented, contents, settings, provenance)


@pytest.mark.parametrize("preset", ["gentle", "all"])
def test_augmentation_missing_variant_aborts_export(store, monkeypatch, preset):
    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    monkeypatch.setattr("labelpilot.exporting.augmented_variants", lambda *args, **kwargs: iter(()))
    with pytest.raises(Problem, match="副本数量缺失"):
        export_project(store, "test", {"split": {"train": 100, "val": 0, "test": 0},
                                        "augmentation": {"preset": preset}})
    assert not list((store.home / "exports").glob("*.zip"))


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("seed", [0, 1, 42, 4294967295])
def test_augmented_raster_objects_remain_aligned_with_boxes(tmp_path, preset, seed):
    source = tmp_path / "asymmetric.png"
    with Image.new("RGB", (101, 79), "black") as image:
        image.paste("white", (13, 11, 41, 32))
        image.paste("white", (57, 45, 93, 71))
        image.save(source)
    document = {"imageWidth": 101, "imageHeight": 79, "shapes": [
        {"label": "Raw_Label-1", "shape_type": "rectangle", "points": [[13, 11], [41, 32]]},
        {"label": "Raw_Label-2", "shape_type": "rectangle", "points": [[57, 45], [93, 45], [93, 71], [57, 71]]}]}
    before = copy.deepcopy(document)
    settings = plan_augmentation({"split": {}, "augmentation": {"preset": preset, "copies": 3, "seed": seed}})
    for _, contents, augmented, provenance in augmented_variants(source, document, "asymmetric", digest(source), settings):
        assert validate_augmented_sample(document, augmented, contents, settings, provenance) == 2
        assert [shape["label"] for shape in augmented["shapes"]] == ["Raw_Label-1", "Raw_Label-2"]
        with Image.open(io.BytesIO(contents)) as image:
            mask = image.convert("L").point(lambda value: 255 if value > 100 else 0)
            for shape in augmented["shapes"]:
                xs, ys = zip(*shape["points"])
                left, upper = max(0, min(xs) - 4), max(0, min(ys) - 4)
                right, lower = min(image.width, max(xs) + 4), min(image.height, max(ys) + 4)
                raster_bounds = mask.crop((left, upper, right, lower)).getbbox()
                assert raster_bounds is not None
                raster_bounds = [raster_bounds[0] + left, raster_bounds[1] + upper,
                                 raster_bounds[2] + left, raster_bounds[3] + upper]
                expected = [min(xs), min(ys), max(xs), max(ys)]
                assert max(abs(actual - target) for actual, target in zip(raster_bounds, expected)) <= 1
    assert document == before


def test_augmentation_catalog_api(store):
    with TestClient(create_app(store.home)) as client:
        response = client.get("/api/augmentation-presets")
        assert response.status_code == 200
        catalog = response.json()["presets"]
        assert {preset["id"] for preset in catalog} == {*PRESETS, "all"}
        assert all(preset["preset_count"] == (len(PRESETS) if preset["id"] == "all" else 1) for preset in catalog)
        assert response.json()["max_copies"] == 3
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_augmentation_transparency_uses_documented_white_background(tmp_path, mode):
    source = tmp_path / "transparent.png"
    with Image.new("RGBA", (20, 10), (255, 0, 0, 0)) as image:
        if mode == "P":
            image = image.convert("P")
        image.save(source)
    document = {"imageWidth": 20, "imageHeight": 10, "shapes": []}
    settings = plan_augmentation({"split": {}, "augmentation": {"preset": "horizontal"}})
    variant = next(augmented_variants(source, document, "transparent", digest(source), settings))
    with Image.open(io.BytesIO(variant[1])) as image:
        assert min(image.getpixel((5, 5))) > 200
    assert variant[2]["shapes"] == [] and not variant[2]["checked"]


def test_augmented_yolo_still_rejects_polygons(store):
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"][0].update(shape_type="polygon", points=[[10, 10], [80, 10], [80, 50]])
    store.save("test", item(store, "human")["id"], data)
    with pytest.raises(Problem, match="多边形"):
        export_project(store, "test", {"split": {"train": 100, "val": 0, "test": 0},
                                        "augmentation": {"preset": "horizontal"}})
    assert not list((store.home / "exports").glob("*.zip"))


@pytest.mark.parametrize("ratios,count,expected", [
    ((80, 10, 10), 3, (1, 1, 1)), ((99, .5, .5), 4, (2, 1, 1)),
    ((33.4, 33.3, 33.3), 10, (4, 3, 3)), ((80, 20, 0), 4, (3, 1, 0)),
    ((100, 0, 0), 1, (1, 0, 0)), ((80, 0, 20), 4, (3, 0, 1))])
def test_split_rounding_and_zero_subsets(ratios, count, expected):
    rows = [{"id": str(index)} for index in range(count)]
    assignments, _, counts = plan_splits(rows, {"split": dict(zip(("train", "val", "test"), ratios))})
    assert tuple(counts.values()) == expected
    assert len(assignments) == count


@pytest.mark.parametrize("settings", [
    False, [], {}, {"train": 80, "val": 10}, {"train": 90, "val": 10, "test": 10},
    {"train": 0, "val": 50, "test": 50}, {"train": -1, "val": 91, "test": 10},
    {"train": "80", "val": 10, "test": 10}, {"train": True, "val": 89, "test": 10},
    {"train": float("nan"), "val": 10, "test": 10}, {"train": float("inf"), "val": 10, "test": 10},
    {"train": 10 ** 400, "val": 10, "test": 10},
    *({"train": 80, "val": 10, "test": 10, "seed": seed} for seed in (True, -1, 1.5, "42", 4294967296)),
    {"train": 80, "val": 10, "test": 10, "unknown": 1}])
def test_invalid_split_rejected_before_creating_export(reviewed_store, settings):
    with pytest.raises(Problem) as error:
        export_project(reviewed_store, "test", {"split": settings})
    assert error.value.status == 422
    assert not (reviewed_store.home / "exports").exists()


def test_split_rejects_too_few_images(store):
    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    with pytest.raises(Problem, match="图片数量不足"):
        export_project(store, "test", {"split": {"train": 80, "val": 10, "test": 10}})
    assert not (store.home / "exports").exists()


def test_split_export_can_omit_test_set(reviewed_store):
    result = export_project(reviewed_store, "test", {"split": {"train": 80, "val": 20, "test": 0}})
    with zipfile.ZipFile(reviewed_store.home / "exports" / result["filename"]) as archive:
        assert "test: null\n" in archive.read("dataset.yaml").decode()
        assert not any("/test/" in name for name in archive.namelist())


def test_export_preview_matches_manifest_and_rejects_changed_revision(reviewed_store):
    store = reviewed_store
    store.add_label("test", "unused")
    options = {"split": {"train": 50, "val": 25, "test": 25}}
    preview = preview_export(store, "test", options)
    assert (preview["images"], preview["annotations"], preview["confirmed_empty"]) == (4, 2, 2)
    classes = {row["name"]: row for row in preview["class_counts"]}
    assert classes["object"]["total"] == 2 and classes["unused"]["splits"] == {"train": 0, "val": 0, "test": 0}
    assert not (store.home / "exports").exists()

    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    with pytest.raises(Problem, match="数据已变化") as error:
        export_project(store, "test", {**options, "preview_token": preview["preview_token"]})
    assert error.value.status == 409
    assert not (store.home / "exports").exists()

    preview = preview_export(store, "test", options)
    result = export_project(store, "test", {**options, "preview_token": preview["preview_token"]})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        assert manifest["split_counts"] == preview["split_counts"]
        expected_counts = dict.fromkeys(manifest["split_counts"], 0)
        for name in archive.namelist():
            if name.startswith("labels/"):
                expected_counts[PurePosixPath(name).parts[1]] += len(archive.read(name).splitlines())
        assert next(row["splits"] for row in preview["class_counts"] if row["name"] == "object") == expected_counts


def test_export_preview_distinguishes_pending_empty_and_pins_external_annotation(store, dataset):
    store.save("test", item(store, "empty")["id"], payload(store, "empty"))
    options = {"reviewed_only": False}
    preview = preview_export(store, "test", options)
    assert (preview["images"], preview["excluded"], preview["unreviewed"]) == (3, 1, 3)
    assert (preview["confirmed_empty"], preview["unreviewed_empty"]) == (0, 1)
    source = dataset[2] / "human.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Problem, match="数据已变化"):
        export_project(store, "test", {**options, "preview_token": preview["preview_token"]})


def test_import_is_not_independent_acceptance(store):
    counts = store.listing("test")["counts"]
    assert counts == {"unstarted": 2, "imported": 1, "predicted": 1, "draft": 0, "reviewed": 0, "skipped": 0, "error": 0}


def test_save_preserves_source_and_attributes(store, dataset):
    source = dataset[2] / "human.json"
    fingerprint = digest(source)
    data = payload(store, "human")
    original = copy.deepcopy(data["document"])
    saved = store.save("test", item(store, "human")["id"], data)
    assert saved["version"] == 1
    assert saved["document"]["shapes"] == original["shapes"]
    assert saved["document"]["flags"] == original["flags"]
    assert saved["document"]["custom_top_level"] == original["custom_top_level"]
    assert digest(source) == fingerprint
    assert not saved["document"]["checked"]


def test_version_conflict(store):
    data = payload(store, "human")
    store.save("test", item(store, "human")["id"], data)
    with pytest.raises(Problem, match="其他窗口"):
        store.save("test", item(store, "human")["id"], data)


def test_external_source_change_blocks_first_save(store, dataset):
    data = payload(store, "human")
    source = dataset[2] / "human.json"
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(Problem, match="其他工具"):
        store.save("test", item(store, "human")["id"], data)


def test_changed_image_rejected(store, dataset):
    data = payload(store, "human")
    Image.new("RGB", (240, 120), "red").save(dataset[1] / "human.jpg")
    with pytest.raises(Problem, match="指纹"):
        store.save("test", item(store, "human")["id"], data)


@pytest.mark.parametrize("point", [[-1, 10], [250, 10], [10, float("nan")], [True, 10]])
def test_invalid_coordinates_rejected(store, point):
    data = payload(store, "human")
    data["document"]["shapes"][0]["points"][0] = point
    with pytest.raises(Problem):
        store.save("test", item(store, "human")["id"], data)


def test_unknown_class_rejected(store):
    data = payload(store, "human")
    data["document"]["shapes"][0]["label"] = "unknown"
    with pytest.raises(Problem, match="字典"):
        store.save("test", item(store, "human")["id"], data)


def test_empty_not_automatic_negative(store):
    data = payload(store, "empty", "reviewed")
    with pytest.raises(Problem, match="负样本"):
        store.save("test", item(store, "empty")["id"], data)
    data["confirmed_empty"] = True
    saved = store.save("test", item(store, "empty")["id"], data)
    assert saved["document"]["labelpilot"]["negative_sample_confirmed"]
    assert saved["status"] == "reviewed"


def test_history_restore_creates_new_draft(store):
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"][0]["description"] = "edited"
    store.save("test", item(store, "human")["id"], data)
    restored = store.restore("test", item(store, "human")["id"], 0, 1)
    assert restored["version"] == 2 and restored["status"] == "draft"
    assert restored["document"]["shapes"][0]["description"] == "readability=unreadable"
    assert [row["version"] for row in store.history("test", item(store, "human")["id"])] == [2, 1, 0]


@pytest.mark.parametrize("name", ["human", "model"])
def test_prediction_cannot_overwrite_imported_annotations(store, name):
    data = payload(store, name, "predicted")
    with pytest.raises(Problem, match="覆盖"):
        store.save("test", item(store, name)["id"], data, predicted=True)


def test_reviewed_only_export_and_numeric_roundtrip(store):
    with pytest.raises(Problem, match="没有"):
        export_project(store, "test", {"format": "yolo"})
    data = payload(store, "human", "reviewed")
    store.save("test", item(store, "human")["id"], data)
    result = export_project(store, "test", {"format": "yolo"})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        assert len(manifest["items"]) == 1
        label = archive.read(next(name for name in archive.namelist() if name.startswith("labels/"))).decode().split()
        center_x, center_y, width, height = map(float, label[1:])
        assert abs((center_x - width / 2) * 240 - 12.5) < 1e-8
        assert abs((center_y - height / 2) * 120 - 10) < 1e-8
        assert archive.read("classes.txt") == b"object\n"


@pytest.mark.parametrize("format_name", ["labelme", "coco"])
def test_other_export_formats(store, format_name):
    data = payload(store, "human", "reviewed")
    points = [[110, 70], [12.5, 70], [12.5, 10], [110, 10]]
    data["document"]["shapes"][0]["points"] = points
    item_id = item(store, "human")["id"]
    store.save("test", item_id, data)
    result = export_project(store, "test", {"format": format_name, "reviewed_only": False, "include_images": False})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        assert not any(name.startswith("images/") for name in archive.namelist())
        if format_name == "labelme":
            document = json.loads(archive.read(next(name for name in archive.namelist() if name.startswith("annotations/"))))
            assert document["shapes"][0]["group_id"] == 9
            assert document["shapes"][0]["points"] == [[12.5, 10], [110, 70]]
            assert document["imagePath"].startswith("../images/")
        else:
            document = json.loads(archive.read("annotations.json"))
            assert len(document["images"]) == 2
            assert document["annotations"][0]["bbox"] == [12.5, 10, 97.5, 60]
    assert store.detail("test", item_id)["document"]["shapes"][0]["points"] == points


def test_polygon_not_silently_converted_for_yolo(store):
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"][0].update(shape_type="polygon", points=[[10, 10], [100, 10], [50, 80]])
    store.save("test", item(store, "human")["id"], data)
    assert "多边形" in preview_export(store, "test", {"format": "yolo"})["blocking_issue"]
    assert preview_export(store, "test", {"format": "coco"})["blocking_issue"] is None
    with pytest.raises(Problem, match="多边形"):
        export_project(store, "test", {"format": "yolo"})
    assert not list((store.home / "exports").glob("*"))


def test_project_registration_and_duplicate_names(store, dataset):
    registered = store.register({"name": "Reusable", "image_root": str(dataset[1]), "classes": ["object"]})
    assert registered["total"] == 4
    assert registered["id"] in Store(store.home).projects
    with pytest.raises(Problem, match="独立"):
        store.register({"name": "Unsafe", "image_root": str(store.home), "classes": ["object"]})


def test_selected_image_files_persist_rescan_and_preserve_annotations(store, dataset):
    images, native = dataset[1:3]
    nested = images / "中文图片"
    nested.mkdir()
    extra = nested / "New_Target.PNG"
    Image.new("RGB", (240, 120)).save(extra)
    sources = {path: digest(path) for folder in dataset[1:] for path in folder.rglob("*") if path.is_file()}
    before = store.detail("test", item(store, "human")["id"])
    registered = store.register({"name": "Selected", "image_root": str(images),
                                 "image_files": [str(images / "human.jpg"), "中文图片/New_Target.PNG"],
                                 "annotation_root": str(native), "classes": ["object", "New_Target"]})
    project_id = registered["id"]
    assert registered["total"] == 2
    selected = store.project(project_id)
    assert {entry["relative"] for entry in selected.items} == {"human.jpg", "中文图片/New_Target.PNG"}
    assert set(selected.config["image_files"]) == {"human.jpg", "中文图片/New_Target.PNG"}
    human = next(entry for entry in selected.items if entry["photo_id"] == "human")
    current = store.detail(project_id, human["id"])
    assert current["status"] == "imported"
    saved = store.save(project_id, human["id"], {**current, "expected_version": current["version"], "status": "reviewed"})
    Image.new("RGB", (240, 120)).save(images / "not_selected.jpg")
    assert store.rescan(project_id)["total"] == 2
    reloaded = Store(store.home)
    assert reloaded.project(project_id).summary()["total"] == 2
    assert reloaded.detail(project_id, human["id"])["document"] == saved["document"]
    assert reloaded.detail(project_id, human["id"])["status"] == "reviewed"
    assert reloaded.detail("test", item(reloaded, "human")["id"]) == before
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())


@pytest.mark.parametrize("selected", [False, "human.jpg", [], ["human.jpg"] * 10001,
    [False], [""], ["\x00"], ["missing.jpg"], ["."], ["human.jpg", "./human.jpg"]])
def test_invalid_selected_images_do_not_register(store, dataset, selected):
    before = store.registry_path.read_bytes()
    with pytest.raises(Problem):
        store.register({"name": "Invalid", "image_root": str(dataset[1]), "image_files": selected, "classes": ["object"]})
    assert store.registry_path.read_bytes() == before
    assert set(store.projects) == {"test"}


def test_selected_images_reject_outside_unsupported_and_manifest_conflict(store, dataset):
    images = dataset[1]
    outside = images.parent / "outside.jpg"
    Image.new("RGB", (240, 120)).save(outside)
    unsupported = images / "notes.txt"
    unsupported.write_text("not an image", encoding="utf8")
    before = store.registry_path.read_bytes()
    for files in ([str(outside)], ["../outside.jpg"], ["notes.txt"]):
        with pytest.raises(Problem):
            store.register({"name": "Invalid", "image_root": str(images), "image_files": files, "classes": ["object"]})
    with pytest.raises(Problem, match="任务清单"):
        store.register({"name": "Conflict", "image_root": str(images), "image_files": ["human.jpg"],
                        "manifest_path": str(images / "manifest.json"), "classes": ["object"]})
    assert store.registry_path.read_bytes() == before


def test_image_browser_api_filters_and_preserves_source_without_existing_project(dataset):
    home, images, native, predictions = dataset
    folder = images / "中文目录"
    folder.mkdir()
    Image.new("RGB", (30, 20)).save(folder / "选择.PNG")
    (images / "ignore.pt").write_bytes(b"do not load")
    (images / "ignore.txt").write_text("not an image", encoding="utf8")
    (home / "projects.json").write_text("[]", encoding="utf8")
    app = create_app(home)
    sources = {path: digest(path) for root in (images, native, predictions) for path in root.rglob("*") if path.is_file()}
    with TestClient(app) as client:
        headers = {"X-LabelPilot-Token": app.state.token}
        assert client.post("/api/images/browse", json={"path": str(images)}).status_code == 403
        assert client.post("/api/images/browse", json={"path": str(images)}, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        listing = client.post("/api/images/browse", json={"path": str(images)}, headers=headers).json()
        assert listing["path"] == str(images)
        assert listing["parent"] == str(images.parent)
        assert listing["directories"] == [{"name": "中文目录", "path": str(folder)}]
        assert {entry["name"] for entry in listing["files"]} == {"human.jpg", "model.jpg", "empty.jpg", "another.jpg"}
        assert listing["roots"] and not listing["partial"]
        nested = client.post("/api/images/browse", json={"path": str(folder)}, headers=headers).json()
        assert [entry["name"] for entry in nested["files"]] == ["选择.PNG"]
        assert client.post("/api/images/browse", json={}, headers=headers).status_code == 200
        for value in ("", "relative/path", False, [], str(images / "human.jpg"), str(images / "missing")):
            assert client.post("/api/images/browse", json={"path": value}, headers=headers).status_code == 422
        response = client.post("/api/projects", json={"name": "Chosen", "image_root": str(folder),
                                                     "image_files": ["选择.PNG"], "classes": ["Raw_Label"]}, headers=headers)
        assert response.status_code == 200 and response.json()["total"] == 1
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())


def test_native_image_picker_api_security_and_selection(dataset, monkeypatch):
    home, images, native, predictions = dataset
    chosen = images / "中文 选择.PNG"
    Image.new("RGB", (24, 12)).save(chosen)
    app = create_app(home)
    before = {path: digest(path) for root in (home, images, native, predictions)
              for path in root.rglob("*") if path.is_file()}
    calls = []

    def run(command, **options):
        calls.append((command, options))
        selected = [str(chosen), str(images / "human.jpg")]
        return native_picker.subprocess.CompletedProcess(command, 0, json.dumps(selected, ensure_ascii=False), "")

    monkeypatch.setattr(native_picker.subprocess, "run", run)
    with TestClient(app) as client:
        headers = {"X-LabelPilot-Token": app.state.token}
        data = {"mode": "file", "path": str(images)}
        assert client.post("/api/images/select", json=data).status_code == 403
        assert client.post("/api/images/select", json=data,
                           headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        assert not calls
        response = client.post("/api/images/select", json=data, headers=headers)
        assert response.status_code == 200
        assert response.json() == {"cancelled": False, "image_root": str(images),
                                   "image_files": [chosen.name, "human.jpg"]}
        for invalid in ({}, {"mode": "other"}, {"mode": "file", "path": []},
                        {"mode": "file", "path": "bad\x00path"}):
            assert client.post("/api/images/select", json=invalid, headers=headers).status_code == 422
    assert len(calls) == 1
    command, options = calls[0]
    assert command[-2:] == ["-m", "labelpilot.native_picker"]
    assert json.loads(options["input"]) == {"mode": "file", "path": str(images)}
    assert options["encoding"] == "utf-8" and options["timeout"] == native_picker.PICKER_TIMEOUT
    assert options["capture_output"] and options["cwd"] == Path(native_picker.__file__).resolve().parent.parent
    assert all(digest(path) == fingerprint for path, fingerprint in before.items())


@pytest.mark.parametrize("mode,selected", [("file", []), ("folder", []), ("folder", ["folder"])])
def test_native_image_picker_folder_and_cancel(dataset, monkeypatch, mode, selected):
    images = dataset[1]
    paths = [str(images)] if selected else []
    monkeypatch.setattr(native_picker.subprocess, "run", lambda *args, **options:
                        native_picker.subprocess.CompletedProcess(args[0], 0, json.dumps(paths), ""))
    expected = {"cancelled": False, "image_root": str(images), "image_files": []} if selected else {"cancelled": True}
    assert native_picker.select_images(mode, str(images / "missing")) == expected
    assert not native_picker.PICKER_LOCK.locked()


@pytest.mark.parametrize("selected", [None, {}, [123], [""], ["bad\x00path"], ["relative.jpg"],
                                     ["missing.jpg"], ["unsupported.txt"], ["human.jpg", "human.jpg"],
                                     ["human.jpg", "../native/human.json"], ["human.jpg"] * 10001])
def test_native_image_picker_rejects_invalid_results(dataset, monkeypatch, selected):
    images = dataset[1]
    (images / "unsupported.txt").write_text("not an image", encoding="utf8")
    if isinstance(selected, list):
        selected = [str(images / value) if isinstance(value, str) and value and "\x00" not in value
                    and value != "relative.jpg" else value for value in selected]
    monkeypatch.setattr(native_picker.subprocess, "run", lambda *args, **options:
                        native_picker.subprocess.CompletedProcess(args[0], 0, json.dumps(selected), ""))
    with pytest.raises(Problem) as caught:
        native_picker.select_images("file")
    assert caught.value.status == 422
    assert not native_picker.PICKER_LOCK.locked()


@pytest.mark.parametrize("failure,status", [("timeout", 408), ("launch", 503), ("exit", 503), ("json", 503)])
def test_native_image_picker_errors_release_lock(monkeypatch, failure, status):
    def run(command, **options):
        if failure == "timeout":
            raise native_picker.subprocess.TimeoutExpired(command, options["timeout"])
        if failure == "launch":
            raise OSError("fixture")
        return native_picker.subprocess.CompletedProcess(command, 1 if failure == "exit" else 0, "not json", "")

    monkeypatch.setattr(native_picker.subprocess, "run", run)
    with pytest.raises(Problem) as caught:
        native_picker.select_images("folder")
    assert caught.value.status == status
    assert not native_picker.PICKER_LOCK.locked()


def test_native_image_picker_prevents_duplicate_windows(dataset, monkeypatch):
    calls = []
    monkeypatch.setattr(native_picker.subprocess, "run", lambda *args, **options: calls.append(args))
    with native_picker.PICKER_LOCK:
        with pytest.raises(Problem) as caught:
            native_picker.select_images("file")
        assert caught.value.status == 409
        assert not calls and native_picker.PICKER_LOCK.locked()
    assert native_picker.initial_directory(str(dataset[1] / "human.jpg")) == str(dataset[1])
    assert native_picker.initial_directory(str(dataset[1] / "missing")) == str(Path.home())


def test_native_image_picker_rejects_cross_folder_files_and_invalid_folder_result(dataset, monkeypatch):
    images = dataset[1]
    nested = images / "nested"
    nested.mkdir()
    Image.new("RGB", (24, 12)).save(nested / "other.png")
    selected = [str(images / "human.jpg"), str(nested / "other.png")]
    monkeypatch.setattr(native_picker.subprocess, "run", lambda *args, **options:
                        native_picker.subprocess.CompletedProcess(args[0], 0, json.dumps(selected), ""))
    for mode in ("file", "folder"):
        with pytest.raises(Problem) as caught:
            native_picker.select_images(mode)
        assert caught.value.status == 422
    selected[:] = [str(images / "human.jpg")]
    with pytest.raises(Problem) as caught:
        native_picker.select_images("folder")
    assert caught.value.status == 422 and not native_picker.PICKER_LOCK.locked()


def test_original_label_names_not_translated(store, dataset):
    project = store.register({"name": "Raw labels", "image_root": str(dataset[1]), "classes": ["sample_container", "panel"]})
    assert [label["display"] for label in project["classes"]] == ["sample_container", "panel"]


def test_custom_label_appended_and_persistent(store):
    initial_names = list(store.project("test").names)
    added = store.add_label("test", "Custom_Target-v2")
    assert [label["name"] for label in added["classes"]] == initial_names + ["Custom_Target-v2"]
    assert added["classes"][-1]["id"] == len(initial_names)
    data = payload(store, "human")
    data["document"]["shapes"][0]["label"] = "Custom_Target-v2"
    saved = store.save("test", item(store, "human")["id"], data)
    assert saved["document"]["shapes"][0]["label"] == "Custom_Target-v2"
    reloaded = Store(store.home)
    assert reloaded.project("test").names == initial_names + ["Custom_Target-v2"]
    assert reloaded.rescan("test")["classes"][-1]["name"] == "Custom_Target-v2"
    result = export_project(reloaded, "test", {"format": "yolo", "reviewed_only": False})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        assert archive.read("classes.txt").decode().splitlines() == initial_names + ["Custom_Target-v2"]
        exported_labels = [archive.read(path).decode() for path in archive.namelist() if path.startswith("labels/")]
        assert any(line.startswith("1 ") for line in exported_labels)


def test_custom_label_does_not_change_source_class_map(store, dataset):
    source = dataset[2] / "class_map.json"
    source.write_text(json.dumps({"names": {"0": "object"}}), encoding="utf-8")
    config = store.configs[0]
    config["class_map_path"] = str(source)
    store.registry_path.write_text(json.dumps([config]), encoding="utf-8")
    store = Store(store.home)
    fingerprint = digest(source)
    store.add_label("test", "raw_label")
    assert digest(source) == fingerprint
    reloaded = Store(store.home)
    assert reloaded.project("test").names == ["object", "raw_label"]
    assert reloaded.configs[0]["imported_class_map_path"] == str(source)
    assert "class_map_path" not in reloaded.configs[0]


def test_duplicate_custom_label_is_idempotent(store):
    store.add_label("test", "new_label")
    store.add_label("test", "new_label")
    assert store.project("test").names == ["object", "new_label"]


def test_label_file_import_append_only_and_persistent(reviewed_store, dataset):
    store = reviewed_store
    sources = {path: digest(path) for folder in dataset[1:] for path in folder.rglob("*") if path.is_file()}
    with store.connect() as connection:
        before = {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY item")]
                  for table in ("annotations", "revisions")}
    result = store.import_labels("test", "classes.txt", "\ufeffobject\r\nMixed_Name-v2\r\npanel\r\nMixed_Name-v2\r\n\r\n")
    assert result["added"] == 2 and result["skipped"] == 2
    assert [(label["id"], label["name"]) for label in result["project"]["classes"]] == [
        (0, "object"), (1, "Mixed_Name-v2"), (2, "panel")]
    repeated = store.import_labels("test", "classes.txt", "object\nMixed_Name-v2\npanel\n")
    assert repeated["added"] == 0 and repeated["skipped"] == 3
    reloaded = Store(store.home)
    assert reloaded.project("test").names == ["object", "Mixed_Name-v2", "panel"]
    exported = export_project(reloaded, "test", {"format": "yolo"})
    with zipfile.ZipFile(store.home / "exports" / exported["filename"]) as archive:
        assert archive.read("classes.txt").decode() == "object\nMixed_Name-v2\npanel\n"
        for path in archive.namelist():
            if path.startswith("labels/"):
                assert all(line.startswith("0 ") for line in archive.read(path).decode().splitlines())
    with store.connect() as connection:
        after = {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY item")]
                 for table in ("annotations", "revisions")}
    assert before == after
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())


@pytest.mark.parametrize("filename,content", [
    ("classes.TXT", "object\nRaw_Label-v2\nPanel\n"),
    ("classes.txt", "Raw_Label-v2\nobject\nPanel\n"),
    ("class_map.json", '["Raw_Label-v2", "Panel"]'),
    ("class_map.JSON", '\ufeff{"names": ["Raw_Label-v2", "Panel"]}'),
    ("class_map.json", '{"names": {"1": "Panel", "0": "Raw_Label-v2"}}'),
    ("class_map.json", '{"0": "Raw_Label-v2", "1": "Panel"}'),
])
def test_label_file_import_formats(store, filename, content):
    result = store.import_labels("test", filename, content)
    assert result["added"] == 2
    assert store.project("test").names == ["object", "Raw_Label-v2", "Panel"]


@pytest.mark.parametrize("filename,content", [
    ("classes.txt", ""), ("classes.txt", " \n\r\n"), ("classes.txt", None),
    ("classes.txt", "a" * (1024 * 1024 + 1)), ("classes.txt", "valid\n" + "a" * 161),
    ("classes.txt", "valid\ncontrol\x00label"), ("classes.json", '["valid", 12]'),
    ("classes.json", '["valid", ""]'), ("classes.json", '{"names": ["valid", "bad\\nlabel"]}'),
    ("classes.json", '{"names": {"0": "valid", "2": "gap"}}'),
    ("classes.json", '{"names": {"zero": "valid"}}'),
    ("classes.json", '{"names": {"00": "valid"}}'),
    ("classes.json", '{"names": {"-1": "valid"}}'),
    ("classes.json", '{"names": []}'), ("classes.json", '{"names": "valid"}'),
    ("classes.json", 'null'), ("classes.json", '["valid",'),
    ("classes.yaml", "valid"), (None, "valid"),
    ("classes.txt", "\ud800"), ("classes.json", '["valid", "\\ud800"]'),
], ids=["empty", "whitespace", "non_text", "oversized", "long_label", "control_character",
        "non_text_label", "empty_label", "newline_label", "id_gap", "non_numeric_id", "padded_id",
        "negative_id", "empty_names", "non_list_names", "null_json", "malformed_json", "unsupported_type",
        "missing_filename", "invalid_utf8", "invalid_json_unicode"])
def test_invalid_label_file_import_is_atomic(store, filename, content):
    registry = store.registry_path.read_bytes()
    with pytest.raises(Problem) as caught:
        store.import_labels("test", filename, content)
    assert caught.value.status == 422
    assert store.registry_path.read_bytes() == registry
    assert store.project("test").names == ["object"]


def test_label_file_import_preserves_source_mapping_and_other_project(store, dataset):
    source = dataset[2] / "class_map.json"
    source.write_text(json.dumps({"names": {"0": "object"}}), encoding="utf-8")
    store.configs[0]["class_map_path"] = str(source)
    store.registry_path.write_text(json.dumps(store.configs), encoding="utf-8")
    store = Store(store.home)
    other = store.register({"name": "Other", "image_root": str(dataset[1]), "classes": ["Other_Label"]})
    fingerprint = digest(source)
    store.import_labels("test", "classes.txt", "new_label\n")
    reloaded = Store(store.home)
    assert digest(source) == fingerprint
    assert reloaded.project("test").names == ["object", "new_label"]
    assert reloaded.configs[0]["imported_class_map_path"] == str(source)
    assert "class_map_path" not in reloaded.configs[0]
    assert reloaded.project(other["id"]).names == ["Other_Label"]


def test_label_file_import_reuses_renamed_label(renamed_store):
    result = renamed_store.import_labels("test", "classes.txt", "object\nobject_1\nNew_Label\n")
    assert result["added"] == 1 and result["skipped"] == 2
    assert renamed_store.project("test").names == ["object_1", "New_Label"]


def test_label_file_import_http_security_and_validation(dataset):
    app = create_app(dataset[0])
    with TestClient(app) as client:
        path = "/api/projects/test/labels/import"
        data = {"filename": "classes.txt", "content": "object\nImported_Label\n"}
        headers = {"X-LabelPilot-Token": app.state.token}
        assert client.post(path, json=data).status_code == 403
        assert client.post(path, json=data, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        assert client.post(path, json={}, headers=headers).status_code == 422
        assert client.post("/api/projects/missing/labels/import", json=data, headers=headers).status_code == 404
        imported = client.post(path, json=data, headers=headers)
        assert imported.status_code == 200
        assert imported.json()["added"] == 1 and imported.json()["skipped"] == 1
        assert imported.json()["project"]["classes"][-1] == {
            "id": 1, "name": "Imported_Label", "display": "Imported_Label", "color": "#38bdf8"}


@pytest.mark.parametrize("name", [None, "", "  ", "new\nlabel", "a" * 161])
def test_invalid_custom_label_rejected(store, name):
    with pytest.raises(Problem):
        store.add_label("test", name)


def test_model_allows_append_only_custom_labels():
    assert model_class_names({0: "object", 1: "panel"}, ["object", "panel", "new_label"]) == ["object", "panel"]
    for project_names in (["panel", "object"], ["object"], ["renamed", "panel"]):
        with pytest.raises(Problem, match="ID"):
            model_class_names({0: "object", 1: "panel"}, project_names)


@pytest.fixture
def renamed_store(store):
    store.save("test", item(store, "human")["id"], payload(store, "human", "reviewed"))
    config = copy.deepcopy(store.configs[0])
    config["classes"] = ["object_1"]
    config["label_aliases"] = {"object": "object_1"}
    store.registry_path.write_text(json.dumps([config]), encoding="utf-8")
    return Store(store.home)


def test_label_alias_reads_legacy_database_and_sources_without_mutation(renamed_store, dataset):
    store = renamed_store
    sources = {path: digest(path) for folder in dataset[2:] for path in folder.rglob("*.json")}
    for name in ("human", "model"):
        current = store.detail("test", item(store, name)["id"])
        assert current["document"]["shapes"][0]["label"] == "object_1"
        assert current["document"]["shapes"][0]["points"] == [[12.5, 10], [110, 70]]
        assert current["document"]["shapes"][0]["group_id"] == 9
    with store.connect() as connection:
        document = json.loads(connection.execute("SELECT document FROM annotations").fetchone()[0])
    assert document["shapes"][0]["label"] == "object"
    assert all(digest(path) == fingerprint for path, fingerprint in sources.items())
    assert store.rescan("test")["classes"][0]["name"] == "object_1"


def test_label_alias_normalizes_stale_browser_save(renamed_store):
    store = renamed_store
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"][0]["label"] = "object"
    before = copy.deepcopy(data)
    saved = store.save("test", item(store, "human")["id"], data)
    assert saved["document"]["shapes"][0] == {**data["document"]["shapes"][0], "label": "object_1"}
    assert data == before
    reloaded = Store(store.home)
    assert reloaded.detail("test", item(reloaded, "human")["id"])["document"]["shapes"][0]["label"] == "object_1"


def test_label_alias_restores_legacy_revision_without_rewriting_history(renamed_store):
    store = renamed_store
    with store.connect() as connection:
        before = [tuple(row) for row in connection.execute("SELECT * FROM revisions ORDER BY version")]
    restored = store.restore("test", item(store, "human")["id"], 0, 1)
    assert restored["version"] == 2 and restored["status"] == "draft"
    assert restored["document"]["shapes"][0]["label"] == "object_1"
    assert restored["document"]["shapes"][0]["description"] == "readability=unreadable"
    with store.connect() as connection:
        after = [tuple(row) for row in connection.execute("SELECT * FROM revisions WHERE version < 2 ORDER BY version")]
    assert after == before


@pytest.mark.parametrize("format_name", ["yolo", "labelme", "coco"])
def test_label_alias_exports_current_name_and_original_id(renamed_store, format_name):
    store = renamed_store
    result = export_project(store, "test", {"format": format_name, "reviewed_only": False})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        assert archive.read("classes.txt") == b"object_1\n"
        if format_name == "yolo":
            lines = [line for path in archive.namelist() if path.startswith("labels/")
                     for line in archive.read(path).decode().splitlines()]
            assert len(lines) == 2 and all(line.startswith("0 ") for line in lines)
        elif format_name == "labelme":
            shapes = [shape for path in archive.namelist() if path.startswith("annotations/")
                      for shape in json.loads(archive.read(path))["shapes"]]
            assert len(shapes) == 2 and all(shape["label"] == "object_1" for shape in shapes)
        else:
            document = json.loads(archive.read("annotations.json"))
            assert document["categories"] == [{"id": 1, "name": "object_1"}]
            assert len(document["annotations"]) == 2
            assert all(shape["category_id"] == 1 for shape in document["annotations"])


def test_label_alias_cannot_be_readded_as_new_class(renamed_store):
    with pytest.raises(Problem, match="已更名为 object_1"):
        renamed_store.add_label("test", "object")
    assert renamed_store.project("test").names == ["object_1"]
    renamed_store.add_label("test", "another_class")
    assert Store(renamed_store.home).project("test").label_aliases == {"object": "object_1"}


def test_label_alias_export_preview_and_clean_preserve_raw_rows(renamed_store):
    store = renamed_store
    with store.connect() as connection:
        before = [tuple(row) for row in connection.execute("SELECT * FROM annotations")]
    preview = preview_export(store, "test", {"format": "labelme", "reviewed_only": False})
    assert preview["blocking_issue"] is None
    assert preview["class_counts"][0]["name"] == "object_1"
    assert preview["annotations"] == 2
    clean = preview_export(store, "test", {"format": "labelme", "clean": True})
    assert clean["blocking_issue"] is None
    assert clean["annotations"] == 1
    assert store.overlay("test", include_documents=True)[item(store, "human")["id"]]["document"]["shapes"][0]["label"] == "object_1"
    with store.connect() as connection:
        assert before == [tuple(row) for row in connection.execute("SELECT * FROM annotations")]


@pytest.mark.parametrize("aliases", [None, [], {"object_1": "other"}, {"old": "missing"},
                                      {"old": "other_old", "other_old": "object_1"},
                                      {"": "object_1"}, {"old": []}])
def test_invalid_label_alias_rejected(store, aliases):
    config = {**store.configs[0], "classes": ["object_1", "other"], "label_aliases": aliases}
    with pytest.raises(Problem, match="别名"):
        Project(config)


def test_model_label_alias_keeps_strict_class_id_validation():
    aliases = {"object": "object_1"}
    names = ["object_1", "panel", "new_label"]
    assert model_class_names({0: "object", 1: "panel"}, names, aliases) == names[:2]
    assert model_class_names({0: "object_1", 1: "panel"}, names, aliases) == names[:2]
    for model_names in ({0: "panel", 1: "object"}, {0: "unknown", 1: "panel"},
                        {0: "object", 1: "object"}, {}):
        with pytest.raises(Problem, match="ID"):
            model_class_names(model_names, names, aliases)


def test_asset_versions_change_with_frontend_content(dataset, tmp_path, monkeypatch):
    assets = tmp_path / "frontend"
    assets.mkdir()
    for name in ("index.html", "efficiency.js", "resume.js", "automation.js", "task-tools.js",
                 "app.js", "style.css", "realityloop.png"):
        (assets / name).write_bytes((server.ASSETS / name).read_bytes())
    monkeypatch.setattr(server, "ASSETS", assets)
    with TestClient(create_app(dataset[0])) as client:
        page = client.get("/")
        assert page.headers["cache-control"] == "no-store"
        assert "__ASSET_VERSION__" not in page.text
        version = re.search(r'/static/app.js\?v=([a-f0-9]{16})', page.text).group(1)
        assert f'/static/style.css?v={version}' in page.text
        assert f'/static/efficiency.js?v={version}' in page.text
        assert version in client.get("/").text
        for name in ("efficiency.js", "resume.js", "automation.js", "app.js", "style.css"):
            for suffix in ("", f"?v={version}"):
                response = client.get(f"/static/{name}{suffix}")
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-store"
            with (assets / name).open("a", encoding="utf-8") as stream:
                stream.write("\n ")
            updated = client.get("/").text
            new_version = re.search(r'/static/app.js\?v=([a-f0-9]{16})', updated).group(1)
            assert new_version != version
            assert f'/static/style.css?v={new_version}' in updated
            assert f'/static/efficiency.js?v={new_version}' in updated
            version = new_version


def test_http_local_security_and_save(dataset):
    app = create_app(dataset[0])
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        page = client.get("/").text
        token = re.search(r'name="labelpilot-token" content="([^"]+)"', page).group(1)
        assert client.post("/api/projects", json={}).status_code == 403
        assert client.get("/api/projects", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/api/projects", headers={"Host": "evil.example"}).status_code == 400
        preview_path = "/api/projects/test/export/preview"
        assert client.post(preview_path, json={}).status_code == 403
        preview = client.post(preview_path, json={"reviewed_only": False}, headers={"X-LabelPilot-Token": token})
        assert preview.status_code == 200 and preview.json()["annotations"] == 2
        store = app.state.store
        row = item(store, "human")
        assert client.get(f"/api/projects/test/items/{row['id']}/image?thumbnail=true").headers["content-type"] == "image/jpeg"
        response = client.put(f"/api/projects/test/items/{row['id']}", json=payload(store, "human"), headers={"X-LabelPilot-Token": token})
        assert response.status_code == 200 and response.json()["version"] == 1
        assert client.post("/api/projects/test/labels", json={"name": "raw_label"}).status_code == 403
        added = client.post("/api/projects/test/labels", json={"name": "raw_label"}, headers={"X-LabelPilot-Token": token})
        assert added.status_code == 200
        assert added.json()["classes"][-1]["name"] == "raw_label"


def test_missing_model_does_not_download(store):
    inference = Inference(store)
    assert Path(os.environ["YOLO_CONFIG_DIR"]).is_relative_to(store.data)
    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["YOLO_OFFLINE"] == "true"
    with pytest.raises(Problem, match="联网下载"):
        inference.predict("test", item(store, "empty")["id"], {})


def test_rescan_keeps_edits_and_finds_new_images(store, dataset):
    store.save("test", item(store, "human")["id"], payload(store, "human"))
    Image.new("RGB", (200, 100)).save(dataset[1] / "new_image.jpg")
    summary = store.rescan("test")
    assert summary["total"] == 5
    assert store.detail("test", item(store, "human")["id"])["version"] == 1


def test_review_updates_preannotation_state_without_editing_source(store, dataset):
    source = dataset[3] / "model.json"
    original = digest(source)
    saved = store.save("test", item(store, "model")["id"], payload(store, "model", "reviewed"))
    assert saved["document"]["preannotation"]["human_reviewed"]
    assert saved["document"]["labelpilot"]["human_reviewed"]
    assert digest(source) == original


def test_zero_area_polygon_rejected(store):
    data = payload(store, "human")
    data["document"]["shapes"][0].update(shape_type="polygon", points=[[10, 10], [20, 20], [30, 30]])
    with pytest.raises(Problem, match="多边形面积"):
        store.save("test", item(store, "human")["id"], data)


def test_batch_processes_only_unstarted_images(store):
    class FakeInference(Inference):
        def predict(self, project_id, item_id, options):
            return {"shapes": [], "provenance": {"human_reviewed": False}}
    jobs = Jobs(store, FakeInference(store))
    job = jobs.start("test", {"limit": 10})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = jobs.list("test")[0]
        if record["status"] not in ("running", "queued"):
            break
        time.sleep(.02)
    assert record["status"] == "completed" and record["done"] == 2
    assert store.listing("test")["counts"]["imported"] == 1
    assert store.listing("test")["counts"]["reviewed"] == 0
    assert job["total"] == 2


def test_model_browser_filters_and_searches_without_loading(dataset):
    folder = dataset[0] / "模型目录"
    nested = folder / "run" / "weights"
    nested.mkdir(parents=True)
    (folder / "best.PT").write_bytes(b"not executable weights")
    (folder / "ignore.txt").write_text("private unrelated content")
    (nested / "final.pt").write_bytes(b"nested model")
    listing = model_files.browse(str(folder))
    assert [row["name"] for row in listing["files"]] == ["best.PT"]
    assert [row["name"] for row in listing["directories"]] == ["run"]
    assert listing["parent"] == str(folder.parent)
    assert listing["roots"] and not listing["partial"]
    found = model_files.search(str(folder))
    assert {row["path"] for row in found["files"]} == {str(folder / "best.PT"), str(nested / "final.pt")}
    assert found["search"] and not found["partial"]
    assert "content" not in json.dumps(found)


@pytest.mark.parametrize("value", [None, "", "relative/path", 12, "bad\x00path"])
def test_model_browser_rejects_invalid_paths(value):
    with pytest.raises(Problem):
        model_files.browse(value)


def test_model_browser_rejects_missing_or_file_paths(dataset):
    for path in (dataset[0] / "missing", dataset[0] / "projects.json"):
        with pytest.raises(Problem):
            model_files.search(str(path))


def test_model_browser_scan_limits_are_visible(dataset, monkeypatch):
    folder = dataset[0] / "models"
    folder.mkdir()
    for name in ("a.pt", "b.pt", "c.pt"):
        (folder / name).write_bytes(b"fixture")
    monkeypatch.setattr(model_files, "MAX_MODELS", 1)
    found = model_files.search(str(folder))
    assert len(found["files"]) == 1 and found["partial"]
    monkeypatch.setattr(model_files, "MAX_ENTRIES", 1)
    listing = model_files.browse(str(folder))
    assert len(listing["files"]) == 1 and listing["partial"]


def test_model_browser_skips_links(dataset):
    folder = dataset[0] / "models"
    folder.mkdir()
    outside = dataset[0] / "outside"
    outside.mkdir()
    (outside / "external.pt").write_bytes(b"fixture")
    try:
        (folder / "shortcut").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires permission on this Windows installation")
    assert model_files.search(str(folder))["files"] == []
    assert model_files.browse(str(folder))["directories"] == []


def test_model_picker_http_security_and_save(dataset, monkeypatch):
    app = create_app(dataset[0])
    weights = dataset[0] / "models" / "weights"
    weights.mkdir(parents=True)
    first = weights / "best.pt"
    second = weights / "final.pt"
    first.write_bytes(b"fixture one; must not be loaded during browsing")
    second.write_bytes(b"fixture two; must not be loaded during browsing")
    store = app.state.store
    before = store.detail("test", item(store, "human")["id"])
    source_hash = digest(dataset[2] / "human.json")
    original_config = store.registry_path.read_bytes()
    with TestClient(app) as client:
        headers = {"X-LabelPilot-Token": app.state.token}
        for endpoint in ("browse", "search"):
            address = f"/api/projects/test/model/{endpoint}"
            assert client.post(address, json={"path": str(weights)}).status_code == 403
            assert client.post(address, json={"path": str(weights)}, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
            assert client.post(address, json={"path": str(weights)}, headers=headers).status_code == 200
        assert store.registry_path.read_bytes() == original_config
        assert client.post("/api/projects/unknown/model/browse", json={}, headers=headers).status_code == 404
        for path in (first, second):
            response = client.post("/api/projects/test/model", json={"path": str(path)}, headers=headers)
            assert response.status_code == 200 and response.json()["model_path"] == str(path)
        default = client.post("/api/projects/test/model/browse", json={}, headers=headers).json()
        assert default["path"] == str(weights)
        assert client.post("/api/projects/test/model", json={"path": str(weights)}, headers=headers).status_code == 400
        assert Store(store.home).project("test").config["model_path"] == str(second)
        with app.state.inference.lock:
            assert client.post("/api/projects/test/model", json={"path": str(first)}, headers=headers).status_code == 409
        app.state.jobs.records["fixture"] = {"project": "test", "status": "queued"}
        assert client.post("/api/projects/test/model", json={"path": str(first)}, headers=headers).status_code == 409
    assert store.detail("test", item(store, "human")["id"]) == before
    assert digest(dataset[2] / "human.json") == source_hash


def test_model_picker_link_detection():
    class Entry:
        def __init__(self, symlink, attributes):
            self.symlink = symlink
            self.attributes = attributes

        def is_symlink(self):
            return self.symlink

        def stat(self, follow_symlinks):
            return type("Info", (), {"st_file_attributes": self.attributes})()

    assert model_files.linked(Entry(True, 0))
    assert model_files.linked(Entry(False, 0x400))
    assert not model_files.linked(Entry(False, 0))
