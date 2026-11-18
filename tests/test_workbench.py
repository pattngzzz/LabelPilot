import copy
import json
import os
import re
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot.exporting import export_project
from labelpilot import model_files
from labelpilot.inference import Inference, Jobs, model_class_names
from labelpilot.server import create_app
from labelpilot.store import Problem, Store, digest


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
    result = export_project(store, "test", {"format": format_name, "reviewed_only": False, "include_images": False})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        assert not any(name.startswith("images/") for name in archive.namelist())
        if format_name == "labelme":
            document = json.loads(archive.read(next(name for name in archive.namelist() if name.startswith("annotations/"))))
            assert document["shapes"][0]["group_id"] == 9
            assert document["imagePath"].startswith("../images/")
        else:
            document = json.loads(archive.read("annotations.json"))
            assert len(document["images"]) == 2
            assert document["annotations"][0]["bbox"] == [12.5, 10, 97.5, 60]


def test_polygon_not_silently_converted_for_yolo(store):
    data = payload(store, "human", "reviewed")
    data["document"]["shapes"][0].update(shape_type="polygon", points=[[10, 10], [100, 10], [50, 80]])
    store.save("test", item(store, "human")["id"], data)
    with pytest.raises(Problem, match="多边形"):
        export_project(store, "test", {"format": "yolo"})
    assert not list((store.home / "exports").glob("*.zip"))


def test_project_registration_and_duplicate_names(store, dataset):
    registered = store.register({"name": "Reusable", "image_root": str(dataset[1]), "classes": ["object"]})
    assert registered["total"] == 4
    assert registered["id"] in Store(store.home).projects
    with pytest.raises(Problem, match="独立"):
        store.register({"name": "Unsafe", "image_root": str(store.home), "classes": ["object"]})


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


@pytest.mark.parametrize("name", [None, "", "  ", "new\nlabel", "a" * 161])
def test_invalid_custom_label_rejected(store, name):
    with pytest.raises(Problem):
        store.add_label("test", name)


def test_model_allows_append_only_custom_labels():
    assert model_class_names({0: "object", 1: "panel"}, ["object", "panel", "new_label"]) == ["object", "panel"]
    for project_names in (["panel", "object"], ["object"], ["renamed", "panel"]):
        with pytest.raises(Problem, match="ID"):
            model_class_names({0: "object", 1: "panel"}, project_names)


def test_http_local_security_and_save(dataset):
    app = create_app(dataset[0])
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        page = client.get("/").text
        token = re.search(r'name="labelpilot-token" content="([^"]+)"', page).group(1)
        assert client.post("/api/projects", json={}).status_code == 403
        assert client.get("/api/projects", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/api/projects", headers={"Host": "evil.example"}).status_code == 400
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
    class FakeInference:
        def predict(self, project_id, item_id, options):
            return {"shapes": [], "provenance": {"human_reviewed": False}}
    jobs = Jobs(store, FakeInference())
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
