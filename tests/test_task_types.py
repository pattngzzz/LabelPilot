import copy
import hashlib
import io
import json
import math
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
    {"shape_type": "polygon", "points": [[5, 8], [65, 8]]},
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


@pytest.mark.parametrize("points", [
    [[10, 10], [65, 10], [40, 45]],
    [[5, 5], [65, 10], [25, 20], [5, 45]],
    [[10, 10], [55, 8], [70, 25], [55, 48], [10, 40]],
    [[5, 10], [65, 10], [65, 25], [35, 25], [35, 45], [5, 45]],
    [[5, 10], [35, 10], [65, 10], [65, 40], [5, 40]],
    [[40 + 25 * math.cos(index * math.tau / 12), 30 + 20 * math.sin(index * math.tau / 12)] for index in range(12)],
])
def test_ocr_multi_vertex_polygon_saves_and_exports_without_losing_vertices(tmp_path, points):
    store, project, item = project_with_image(tmp_path, "ocr")
    source = Path(project.item(item)["path"]).read_bytes()
    shape = ocr_shape("25.0°C", shape_type="polygon", points=points, value_decimal="25.0", unit="°C")
    reviewed(store, project, item, [shape])
    assert Store(store.home).detail(project.id, item)["document"]["shapes"] == [shape]
    for format_name in ("paddleocr", "labelme"):
        with archive(store, project, format_name) as exported:
            annotation = next(name for name in exported.namelist() if name.startswith("annotations/"))
            assert json.loads(exported.read(annotation))["shapes"] == [shape]
            if format_name == "paddleocr":
                entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
                assert entries == [{"transcription": "25.0°C", "points": points}]
                crop_name = next(name for name in exported.namelist() if name.startswith("recognition/"))
                with exported.open(crop_name) as stream, Image.open(stream) as crop:
                    assert crop.width >= 2 and crop.height >= 2
                assert exported.read("rec/all.txt").decode().endswith("\t25.0°C\n")
    assert Path(project.item(item)["path"]).read_bytes() == source


@pytest.mark.parametrize("points", [
    [[5, 5], [70, 45], [5, 45], [70, 5], [75, 55]],
    [[5, 5], [65, 5], [65, 45], [5, 5], [5, 45]],
    [[5, 5], [65, 5], [35, 5], [65, 45], [5, 45]],
    [[5, 5], [25, 5], [45, 5], [65, 5]],
])
def test_ocr_invalid_multi_vertex_polygon_is_rejected_without_saving(tmp_path, points):
    store, project, item = project_with_image(tmp_path, "ocr")
    with pytest.raises(Problem, match="交叉|重叠|退化|面积"):
        reviewed(store, project, item, [ocr_shape(shape_type="polygon", points=points)])
    assert store.detail(project.id, item)["version"] == 0


def test_ocr_concave_polygon_crop_masks_outside_without_changing_source():
    from labelpilot.ocr import recognition_crop

    source = Image.new("RGB", (80, 60), "red")
    before = source.tobytes()
    crop = recognition_crop(source, [[5, 10], [65, 10], [65, 25], [35, 25], [35, 45], [5, 45]])
    assert crop.getpixel((5, 5)) == (255, 0, 0)
    assert crop.getpixel((50, 25)) == (255, 255, 255)
    assert source.tobytes() == before


@pytest.mark.parametrize("degrees", [-90, -5, -1, 1, 5, 90])
def test_ocr_rotated_box_survives_save_reload_and_exports(tmp_path, degrees):
    store, project, item = project_with_image(tmp_path, "ocr")
    source = Path(project.item(item)["path"]).read_bytes()
    radians = math.radians(degrees)
    points = [[40 + horizontal * math.cos(radians) - vertical * math.sin(radians),
               30 + horizontal * math.sin(radians) + vertical * math.cos(radians)]
              for horizontal, vertical in [(-20, -8), (20, -8), (20, 8), (-20, 8)]]
    shape = ocr_shape("00.250 g", shape_type="rectangle", points=points, rotation=degrees, custom_attribute={"keep": True})
    reviewed(store, project, item, [shape])
    assert Store(store.home).detail(project.id, item)["document"]["shapes"] == [shape]
    with archive(store, project, "paddleocr") as exported:
        entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
        assert entries == [{"transcription": "00.250 g", "points": points}]
        assert exported.read("rec/all.txt").decode().endswith("\t00.250 g\n")
        crop_name = next(name for name in exported.namelist() if name.startswith("recognition/"))
        with exported.open(crop_name) as stream, Image.open(stream) as crop:
            assert 40 <= crop.width <= 41 and 16 <= crop.height <= 17
    with archive(store, project, "labelme") as exported:
        annotation = next(name for name in exported.namelist() if name.startswith("annotations/"))
        assert json.loads(exported.read(annotation))["shapes"] == [{**shape, "shape_type": "polygon"}]
    assert Path(project.item(item)["path"]).read_bytes() == source


@pytest.mark.parametrize("task", ["detect", "ocr"])
def test_outside_rotated_rectangle_keeps_rectangle_type_and_exact_corners(tmp_path, task):
    store, project, item = project_with_image(tmp_path, task, ["text"])
    radians = math.radians(15)
    points = [[40 + horizontal * math.cos(radians) - vertical * math.sin(radians),
               30 + horizontal * math.sin(radians) + vertical * math.cos(radians)]
              for horizontal, vertical in [(-40, -30), (40, -30), (40, 30), (-40, 30)]]
    shape = ocr_shape(shape_type="rectangle", points=points, rotation=15)
    reviewed(store, project, item, [shape])
    assert Store(store.home).detail(project.id, item)["document"]["shapes"] == [shape]
    if task == "ocr":
        with archive(store, project, "paddleocr") as exported:
            entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
            assert entries == [{"transcription": "00.250 g", "points": points}]
    else:
        with archive(store, project, "yolo") as exported:
            name = next(name for name in exported.namelist() if name.startswith("labels/"))
            assert list(map(float, exported.read(name).split())) == pytest.approx([0, .5, .5, 1, 1])
    with archive(store, project, "labelme") as exported:
        annotation = next(name for name in exported.namelist() if name.startswith("annotations/"))
        assert json.loads(exported.read(annotation))["shapes"] == [{**shape, "shape_type": "polygon"}]
    assert store.detail(project.id, item)["document"]["shapes"] == [shape]


@pytest.mark.parametrize("points,rotation", [
    ([[5, 5], [65, 8], [65, 40], [5, 40]], 5),
    ([[5, 5], [65, 40]], 5),
    ([[5, 5], [65, 5], [65, 40], [5, 40]], "5"),
])
def test_rotated_rectangle_cannot_be_saved_as_a_deformed_polygon(tmp_path, points, rotation):
    store, project, item = project_with_image(tmp_path, "ocr")
    with pytest.raises(Problem, match="旋转矩形"):
        reviewed(store, project, item, [ocr_shape(points=points, rotation=rotation)])
    assert store.detail(project.id, item)["version"] == 0


def test_detection_export_of_rotated_box_uses_bounds_only_when_requested(tmp_path):
    store, project, item = project_with_image(tmp_path, "detect", ["part"])
    points = [[30, 10], [60, 30], [50, 45], [20, 25]]
    shape = {"label": "part", "shape_type": "polygon", "points": points}
    reviewed(store, project, item, [shape])
    result = export_project(store, project.id, {"format": "yolo", "output_task": "detect"})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as exported:
        name = next(name for name in exported.namelist() if name.startswith("labels/"))
        values = list(map(float, exported.read(name).split()))
        assert values == pytest.approx([0, .5, 27.5 / 60, .5, 35 / 60])
    assert store.detail(project.id, item)["document"]["shapes"] == [shape]


@pytest.mark.parametrize("task", ["detect", "segment", "ocr"])
def test_outside_rotated_box_saves_and_preserves_full_corners(tmp_path, task):
    store, project, item = project_with_image(tmp_path, task, ["text"])
    points = [[-5, 8], [75, -4], [84, 48], [4, 60]]
    shape = ocr_shape(shape_type="polygon", points=points, rotation=-5)
    reviewed(store, project, item, [shape])
    assert Store(store.home).detail(project.id, item)["document"]["shapes"] == [shape]
    with archive(store, project, "labelme") as exported:
        annotation = next(name for name in exported.namelist() if name.startswith("annotations/"))
        assert json.loads(exported.read(annotation))["shapes"] == [shape]
    if task == "ocr":
        with archive(store, project, "paddleocr") as exported:
            entries = json.loads(exported.read("det/all.txt").decode().strip().split("\t", 1)[1])
            assert entries == [{"transcription": "00.250 g", "points": points}]
            assert exported.read("rec/all.txt").decode().endswith("\t00.250 g\n")
    else:
        result = export_project(store, project.id, {"format": "yolo", "output_task": "detect"})
        with zipfile.ZipFile(store.home / "exports" / result["filename"]) as exported:
            name = next(name for name in exported.namelist() if name.startswith("labels/"))
            assert list(map(float, exported.read(name).split())) == pytest.approx([0, .5, .5, 1, 1])
        with pytest.raises(Problem, match="越界顶点"):
            export_project(store, project.id, {"format": "yolo", "output_task": "segment"})
    assert store.detail(project.id, item)["document"]["shapes"] == [shape]


@pytest.mark.parametrize("rotation", [None, True, "5", float("nan"), float("inf")])
def test_unmarked_or_invalid_rotated_box_still_rejects_outside_coordinates(tmp_path, rotation):
    store, project, item = project_with_image(tmp_path, "ocr")
    shape = ocr_shape(shape_type="polygon", points=[[-5, 8], [75, -4], [84, 48], [4, 60]], rotation=rotation)
    with pytest.raises(Problem, match="超出原图边界"):
        reviewed(store, project, item, [shape])
    assert store.detail(project.id, item)["version"] == 0


@pytest.mark.parametrize("task,points", [
    ("semantic", [[-5, 8], [75, -4], [84, 48], [4, 60]]),
    ("detect", [[-5, 8], [75, -4], [30, 25], [4, 60]]),
    ("detect", [[-5, 8], [75, -4], [84, 48], [4, 60], [0, 20]]),
])
def test_rotation_boundary_exception_does_not_apply_to_masks_or_general_contours(tmp_path, task, points):
    store, project, item = project_with_image(tmp_path, task, ["text"])
    with pytest.raises(Problem, match="超出原图边界"):
        reviewed(store, project, item, [ocr_shape(shape_type="polygon", points=points, rotation=5)])
    assert store.detail(project.id, item)["version"] == 0


@pytest.mark.parametrize("preset", ["gentle", "horizontal", "rotation"])
def test_outside_rotated_box_augmentation_preserves_transformed_corners(tmp_path, preset):
    from labelpilot.augmentation import augmented_variants, plan_augmentation, validate_augmented_sample

    store, project, item = project_with_image(tmp_path, "detect", ["text"])
    points = [[-5, 8], [75, -4], [84, 48], [4, 60]]
    reviewed(store, project, item, [ocr_shape(shape_type="polygon", points=points, rotation=-5)])
    current = store.detail(project.id, item)
    settings = plan_augmentation({"split": {}, "augmentation": {"preset": preset, "copies": 1}})
    for _, contents, augmented, provenance in augmented_variants(
            Path(project.item(item)["path"]), current["document"], item, current["image_sha256"], settings):
        assert validate_augmented_sample(current["document"], augmented, contents, settings, provenance) == 1
        augmented["shapes"][0]["points"][0][0] += 1
        with pytest.raises(Problem, match="标签、坐标或属性不匹配"):
            validate_augmented_sample(current["document"], augmented, contents, settings, provenance)
    assert store.detail(project.id, item)["document"]["shapes"][0]["points"] == points


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


def test_ocr_augmentation_catalog_is_task_specific(tmp_path):
    from fastapi.testclient import TestClient
    from labelpilot.augmentation import OCR_PRESETS, PRESETS
    from labelpilot.server import create_app

    with TestClient(create_app(tmp_path / "workspace")) as client:
        for task, presets in (("ocr", OCR_PRESETS), ("detect", PRESETS), ("segment", PRESETS)):
            response = client.get("/api/augmentation-presets", params={"task_type": task})
            assert response.status_code == 200
            catalog = response.json()["presets"]
            assert {entry["id"] for entry in catalog} == {*presets, "all"}
            assert catalog[-1]["preset_count"] == len(presets)
        assert client.get("/api/augmentation-presets", params={"task_type": "semantic"}).status_code == 422


@pytest.mark.parametrize("task,preset", [("ocr", "horizontal"), ("ocr", "rotation"), ("ocr", "gentle"),
                                         ("detect", "ocr_gentle"), ("segment", "ocr_quality")])
def test_ocr_augmentation_rejects_cross_task_presets(tmp_path, task, preset):
    store, project, item = project_with_image(tmp_path, task, ["text"])
    shapes = [polygon("text", 5, 5, 50, 30)] if task == "segment" else [ocr_shape()]
    reviewed(store, project, item, shapes)
    with pytest.raises(Problem, match="当前任务"):
        preview_export(store, project.id, {"format": "paddleocr" if task == "ocr" else "yolo",
                                         "split": {"train": 100, "val": 0, "test": 0},
                                         "augmentation": {"preset": preset}})


@pytest.mark.parametrize("preset", ["ocr_gentle", "ocr_lighting", "ocr_quality", "all"])
@pytest.mark.parametrize("format_name", ["paddleocr", "labelme"])
def test_ocr_augmentation_preserves_shapes_text_crops_and_original_splits(tmp_path, preset, format_name):
    from labelpilot.ocr import detection_entries, recognition_crop

    store, project, item = project_with_image(tmp_path, "ocr")
    for index in range(3):
        Image.new("RGB", (80, 60), (80 + index * 30, 100, 150)).save(project.root / f"frame{index}.png")
    store.rescan(project.id)
    project = store.project(project.id)
    shapes = [ocr_shape(), ocr_shape("25.0°C", shape_type="polygon", value_decimal="25.0", unit="°C",
                                    points=[[5, 30], [65, 30], [65, 40], [35, 40], [35, 50], [5, 50]]),
              ocr_shape("1.0%", value_decimal="1.0", unit="%", rotation=15,
                        points=[[1, -2], [76, 18], [68, 48], [-7, 28]]),
              ocr_shape("1?3", readability="partial", value_decimal=None),
              ocr_shape(None, readability="unreadable", value_decimal=None)]
    snapshots = {}
    for index, row in enumerate(project.items):
        reviewed(store, project, row["id"], shapes if index else [], confirmed_empty=index == 0)
        snapshots[row["id"]] = store.detail(project.id, row["id"])
    originals = {row["id"]: Path(row["path"]).read_bytes() for row in project.items}
    options = {"format": format_name, "split": {"train": 50, "val": 25, "test": 25, "seed": 42},
               "augmentation": {"preset": preset, "copies": 1, "seed": 123}}
    expected_augmented = 6 if preset == "all" else 2
    preview = preview_export(store, project.id, options)
    assert preview["augmented_images"] == expected_augmented
    result = export_project(store, project.id, options)
    assert result["integrity_check"]["passed"]
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as exported:
        manifest = json.loads(exported.read("export_manifest.json"))
        assert manifest["split_counts"] == {"train": 2 + expected_augmented, "val": 1, "test": 1}
        assert manifest["augmentation"]["output"] == "PNG"
        assert len(manifest["items"]) == 4 + expected_augmented
        originals_by_id = {entry["id"]: entry for entry in manifest["items"] if "source_id" not in entry}
        crop_count = 0
        for entry in manifest["items"]:
            source_id = entry.get("source_id", entry["id"])
            expected = snapshots[source_id]["document"]
            annotation = "annotations/" + str(Path(entry["image"]).relative_to("images").with_suffix(".json")).replace("\\", "/")
            actual = json.loads(exported.read(annotation))
            expected_shapes = copy.deepcopy(expected["shapes"])
            if format_name == "labelme":
                for shape in expected_shapes:
                    if "rotation" in shape:
                        shape["shape_type"] = "polygon"
            assert actual["shapes"] == expected_shapes
            assert (actual["imageWidth"], actual["imageHeight"]) == (80, 60)
            if "source_id" in entry:
                assert entry["split"] == originals_by_id[source_id]["split"] == "train"
                assert entry["image"].endswith(".png") and not entry["human_reviewed"]
                assert not entry["augmentation"]["operations"]["horizontal_flip"]
                assert "rotation_degrees" not in entry["augmentation"]["operations"]
                assert actual["labelpilot"]["status"] == "draft" and not actual["labelpilot"]["human_reviewed"]
            else:
                assert exported.read(entry["image"]) == originals[source_id]
            if format_name == "paddleocr":
                entries = dict(line.split("\t", 1) for line in exported.read(f"det/{entry['split']}.txt").decode().splitlines())
                assert json.loads(entries[entry["image"]]) == detection_entries(expected)
                for index, shape in enumerate(detection_entries(expected), 1):
                    if shape["transcription"] == "###":
                        continue
                    crop_count += 1
                    crop_name = f"recognition/{entry['split']}/{Path(entry['image']).stem}_{index:04d}.png"
                    with Image.open(io.BytesIO(exported.read(entry["image"]))) as image:
                        expected_crop = recognition_crop(image.convert("RGB"), shape["points"])
                    with Image.open(io.BytesIO(exported.read(crop_name))) as crop:
                        assert crop.size == expected_crop.size and crop.tobytes() == expected_crop.tobytes()
        if format_name == "paddleocr":
            labels = [line for subset in ("train", "val", "test")
                      for line in exported.read(f"rec/{subset}.txt").decode().splitlines()]
            assert len(labels) == crop_count
            assert all(line.split("\t", 1)[1] in ("00.250 g", "25.0°C", "1.0%") for line in labels)
    for row in project.items:
        current = store.detail(project.id, row["id"])
        assert current["version"] == snapshots[row["id"]]["version"]
        assert current["document"] == snapshots[row["id"]]["document"]
        assert Path(row["path"]).read_bytes() == originals[row["id"]]


@pytest.mark.parametrize("preset", ["ocr_gentle", "ocr_lighting", "ocr_quality"])
def test_ocr_augmentation_is_reproducible_and_keeps_copy_prefix(tmp_path, preset):
    from labelpilot.augmentation import augmented_variants, plan_augmentation

    store, project, item = project_with_image(tmp_path, "ocr")
    reviewed(store, project, item, [ocr_shape()])
    current = store.detail(project.id, item)
    source = project.item(item)["path"]
    options = {"split": {}, "augmentation": {"preset": preset, "copies": 3, "seed": 42}}
    settings = plan_augmentation(options, "ocr")
    variants = list(augmented_variants(source, current["document"], item, current["image_sha256"], settings))
    assert variants == list(augmented_variants(source, current["document"], item, current["image_sha256"], settings))
    first = next(augmented_variants(source, current["document"], item, current["image_sha256"], {**settings, "copies": 1}))
    assert variants[0] == first
    changed = list(augmented_variants(source, current["document"], item, current["image_sha256"], {**settings, "seed": 43}))
    assert [entry[3] for entry in variants] != [entry[3] for entry in changed]


@pytest.mark.parametrize("corruption", ["text", "points", "missing", "attribute"])
def test_ocr_augmentation_aborts_if_transcription_or_geometry_changes(tmp_path, monkeypatch, corruption):
    import labelpilot.exporting as exporting

    store, project, item = project_with_image(tmp_path, "ocr")
    reviewed(store, project, item, [ocr_shape()])
    generate = exporting.augmented_variants
    def corrupted(*args):
        for index, contents, document, provenance in generate(*args):
            if corruption == "text":
                document["shapes"][0]["text_raw"] = "WRONG"
            elif corruption == "points":
                document["shapes"][0]["points"][0][0] += 1
            elif corruption == "missing":
                document["shapes"].pop()
            else:
                document["shapes"][0]["unit"] = "WRONG"
            yield index, contents, document, provenance
    monkeypatch.setattr(exporting, "augmented_variants", corrupted)
    with pytest.raises(Problem, match="对象数量|标签、坐标或属性"):
        export_project(store, project.id, {"format": "paddleocr", "split": {"train": 100, "val": 0, "test": 0},
                                          "augmentation": {"preset": "all"}})
    assert not list((store.home / "exports").glob("*.zip*"))


def test_ocr_augmentation_cache_and_incremental_updates_preserve_pairing(tmp_path, monkeypatch):
    import labelpilot.exporting as exporting

    store, project, item = project_with_image(tmp_path, "ocr")
    reviewed(store, project, item, [ocr_shape(), ocr_shape("ABC", value_decimal=None)])
    jobs = ExportJobs(store)
    def run(options):
        record = jobs.start(project.id, options)
        jobs.worker.join(timeout=30)
        record = jobs.get(project.id, record["id"])
        assert record["status"] == "completed", record
        assert record["result"]["integrity_check"]["passed"]
        return record
    def artifacts(root):
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in ("images", "annotations", "recognition", "det", "rec")
                for path in (root / directory).rglob("*") if path.is_file()}
    try:
        options = {"format": "paddleocr", "destination": str(tmp_path), "workers": 2,
                   "split": {"train": 100, "val": 0, "test": 0}, "augmentation": {"preset": "all"}}
        cold = run(options)
        root = Path(cold["result"]["directory"])
        original_files = artifacts(root)
        assert cold["cache_misses"] == 3 and cold["result"]["images"] == 4
        with monkeypatch.context() as patch:
            patch.setattr(exporting, "augmented_variants", lambda *args: pytest.fail("Warm cache must not regenerate"))
            warm = run(options)
        assert warm["cache_hits"] == 3 and warm["cache_errors"] == 0
        assert artifacts(Path(warm["result"]["directory"])) == original_files
        options.update(destination=str(root), incremental=True)
        assert run(options)["result"]["incremental"]["skipped"] == 1
        assert artifacts(root) == original_files
        Image.new("RGB", (80, 60), "white").save(project.root / "new.png")
        store.rescan(project.id)
        project = store.project(project.id)
        added = next(row["id"] for row in project.items if row["id"] != item)
        reviewed(store, project, added, [], confirmed_empty=True)
        addition = run(options)["result"]
        assert addition["images"] == 8 and addition["incremental"]["added"] == 1
        assert addition["incremental"]["skipped"] == 1
        reviewed(store, project, item, [ocr_shape("UPDATED", value_decimal=None)])
        assert run(options)["result"]["incremental"]["updated"] == 1
        options["augmentation"]["copies"] = 2
        changed = run(options)["result"]
        assert changed["images"] == 14 and changed["incremental"]["augmentation_changed"]
        lines = (root / "rec" / "train.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 7 and all(line.endswith("\tUPDATED") for line in lines)
        assert len(list((root / "recognition" / "train").glob("*.png"))) == 7
        options.pop("augmentation")
        removed = run(options)["result"]
        assert removed["images"] == 2 and removed["augmented_images"] == 0
        assert len(list((root / "recognition" / "train").glob("*.png"))) == 1
        assert not list((root / "images" / "train").glob("*__aug_*"))
        assert len((root / "det" / "train.txt").read_text(encoding="utf-8").splitlines()) == 2
        assert (root / "rec" / "train.txt").read_text(encoding="utf-8").endswith("\tUPDATED\n")
    finally:
        jobs.shutdown()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("task,format_name", [("ocr", "paddleocr"), ("ocr", "labelme"), ("semantic", "mask"),
                                             ("segment", "yolo"), ("hand_pose", "yolo"), ("hand_pose", "coco")])
def test_incremental_export_task_types_and_ocr_crop_cleanup(tmp_path, task, format_name, legacy):
    store, project, item_id = project_with_image(tmp_path, task, ["text"] if task == "ocr" else ["object"])
    if task == "ocr":
        shapes = [ocr_shape(), ocr_shape("ABC", value_decimal=None)]
    elif task == "hand_pose":
        shapes = [{"label": "object", "shape_type": "hand", "points": [[5, 5], [70, 50]],
                   "keypoints": [[10, 10, 2] for _ in range(HAND_COUNT)]}]
    else:
        shapes = [polygon("object", 5, 5, 50, 30)]
    reviewed(store, project, item_id, shapes)
    jobs = ExportJobs(store)

    def run(options):
        preview = preview_export(store, project.id, options)
        record = jobs.start(project.id, {**options, "preview_token": preview["preview_token"]})
        jobs.worker.join(timeout=20)
        record = jobs.get(project.id, record["id"])
        assert record["status"] == "completed", record.get("error", record)
        return record["result"]

    try:
        options = {"format": format_name, "destination": str(tmp_path)}
        result = run(options)
        root = Path(result["directory"])
        if legacy:
            manifest = json.loads((root / "export_manifest.json").read_bytes())
            manifest.pop("metadata_files")
            for entry in manifest["items"]:
                for key in ("files", "source_stat", "annotation_sha256"):
                    entry.pop(key, None)
            atomic_json(root / "export_manifest.json", manifest)
        options.update(destination=str(root), incremental=True)
        first = run(options)
        assert first["incremental"]["skipped"] == (0 if legacy and format_name in ("coco", "paddleocr") else 1)
        Image.new("RGB", (80, 60), "#123456").save(project.root / "new.png")
        store.rescan(project.id)
        project = store.project(project.id)
        new_id = next(row["id"] for row in project.items if row["photo_id"] == "new")
        reviewed(store, project, new_id, shapes)
        result = run(options)
        assert result["images"] == 2 and result["incremental"]["added"] == 1 and result["incremental"]["skipped"] == 1
        if task == "ocr":
            reviewed(store, project, item_id, [ocr_shape("UPDATED", value_decimal=None)])
            assert run(options)["incremental"]["updated"] == 1
            if format_name == "paddleocr":
                assert len((root / "det" / "all.txt").read_text(encoding="utf-8").splitlines()) == 2
                lines = (root / "rec" / "all.txt").read_text(encoding="utf-8").splitlines()
                assert len(lines) == 3 and sum(line.endswith("\tUPDATED") for line in lines) == 1
                assert len(list((root / "recognition").glob("*.png"))) == 3
                manifest = json.loads((root / "export_manifest.json").read_bytes())
                assert manifest["ocr"]["splits"]["all"] == {"images": 2, "recognition_crops": 3}
        assert run(options)["incremental"]["skipped"] == 2
    finally:
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
