import json
import re
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot import training
from labelpilot.exporting import export_project, preview_export
from labelpilot.server import create_app
from labelpilot.store import Problem, Store, atomic_json


def make_project(tmp_path, samples):
    root = tmp_path / "images"
    root.mkdir()
    manifest = tmp_path / "manifest.json"
    tasks = []
    for name, color, group in samples:
        filename = f"{name}.png"
        Image.new("RGB", (96, 96), color).save(root / filename)
        tasks.append({"package_image": filename, "source_group": group})
    atomic_json(manifest, {"tasks": tasks})
    store = Store(tmp_path / "tool")
    project = store.register({"name": "Workflow test", "image_root": str(root), "manifest_path": str(manifest),
                              "classes": ["object", "other"]})
    return store, project["id"], root


def annotate(store, project_id, photo_id, status, shapes):
    item = next(row for row in store.project(project_id).items if row["photo_id"] == photo_id)
    current = store.detail(project_id, item["id"])
    current["document"]["shapes"] = shapes
    return store.save(project_id, item["id"], {**current, "expected_version": current["version"],
                                             "status": status, "confirmed_empty": True}, predicted=status == "predicted")


def rectangle(**overrides):
    return {"label": "object", "shape_type": "rectangle", "points": [[10, 15], [65, 70]], **overrides}


def test_review_queue_prioritizes_pending_only_and_sorts_before_pagination(tmp_path):
    names = ["easy", "lower", "higher", "difficult", "empty", "tagged_done", "unstarted"]
    store, project, _ = make_project(tmp_path, [(name, (index, 0, 0), "capture") for index, name in enumerate(names)])
    annotate(store, project, "easy", "draft", [rectangle(score=True)])
    annotate(store, project, "lower", "predicted", [rectangle(score=.15)])
    annotate(store, project, "higher", "predicted", [rectangle(score=.48)])
    annotate(store, project, "difficult", "draft", [rectangle(difficult=True, score=.84)])
    annotate(store, project, "empty", "predicted", [])
    annotate(store, project, "tagged_done", "reviewed", [rectangle(difficult=True, score=.02)])

    def queue(status, **pagination):
        return [row["photo_id"] for row in store.listing(project, status=status, **pagination)["items"]]

    assert queue("review", limit=2) == ["empty", "difficult"]
    assert queue("review", offset=2) == ["lower", "higher", "easy"]
    assert queue("review_empty") == ["empty"]
    assert queue("review_difficult") == ["difficult"]
    assert queue("review_low") == ["lower", "higher"]
    annotate(store, project, "empty", "reviewed", [])
    assert queue("review_empty") == []


def test_clean_export_preserves_inputs_and_lists_duplicates_conflicts_and_invalid_images(tmp_path):
    samples = [(name, color, "capture") for name, color in (
        ("alpha", "blue"), ("alpha_copy", "blue"), ("conflict_a", "red"), ("conflict_b", "red"),
        ("negative", "black"), ("broken", "white"), ("pending", "green"),
    )]
    store, project, root = make_project(tmp_path, samples)
    annotate(store, project, "alpha", "reviewed", [rectangle(score=.8)])
    annotate(store, project, "alpha_copy", "reviewed", [rectangle(points=[[65, 70], [65, 15], [10, 15], [10, 70]], score=.9)])
    annotate(store, project, "conflict_a", "reviewed", [rectangle()])
    annotate(store, project, "conflict_b", "reviewed", [rectangle(label="other")])
    annotate(store, project, "negative", "reviewed", [])
    annotate(store, project, "broken", "reviewed", [rectangle()])
    annotate(store, project, "pending", "draft", [rectangle()])
    (root / "broken.png").write_bytes(b"corrupted fixture")
    original = {path.name: path.read_bytes() for path in root.glob("*.png")}
    revisions = store.overlay(project)

    opts = {"clean": True, "format": "labelme"}
    preview = preview_export(store, project, opts)
    assert preview["images"] == 2 and preview["confirmed_empty"] == 1
    assert preview["cleaning"]["counts"] == {"duplicate": 1, "conflict": 2, "unreadable": 1, "annotation": 0}
    result = export_project(store, project, {**opts, "preview_token": preview["preview_token"]})
    with zipfile.ZipFile(store.home / "exports" / result["filename"]) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        assert {item["photo_id"] for item in manifest["items"]} == {"alpha", "negative"}
        assert {item["photo_id"] for item in manifest["cleaning"]} == {"alpha_copy", "conflict_a", "conflict_b", "broken"}
    assert original == {path.name: path.read_bytes() for path in root.glob("*.png")}
    assert revisions == store.overlay(project)


def test_clean_preview_reports_conflicts_when_no_image_can_be_retained(tmp_path):
    store, project, _ = make_project(tmp_path, [("a", "red", "first"), ("b", "red", "second")])
    annotate(store, project, "a", "reviewed", [rectangle()])
    annotate(store, project, "b", "reviewed", [rectangle(label="other")])
    preview = preview_export(store, project, {"clean": True})
    assert not preview["images"] and preview["blocking_issue"]
    assert {row["photo_id"] for row in preview["cleaning"]["examples"]} == {"a", "b"}
    with pytest.raises(Problem, match="没有可用图片"):
        export_project(store, project, {"clean": True, "preview_token": preview["preview_token"]})


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("model_only", [False, True])
def test_training_api_stages_reviewed_snapshot_and_serializes_local_model_tasks(tmp_path, monkeypatch, cancel, renamed, model_only):
    store, project, _ = make_project(tmp_path, [("a", "red", "first"), ("b", "blue", "first"),
                                                ("c", "white", "second"), ("pending", "black", "second")])
    for name in ("a", "b", "c"):
        annotate(store, project, name, "reviewed", [rectangle()])
    if renamed:
        store.configs[0]["classes"][0] = "object_1"
        store.configs[0]["label_aliases"] = {"object": "object_1"}
        atomic_json(store.registry_path, store.configs)
    model_aliases = {"legacy_object": "object_1" if renamed else "object"} if model_only else {}
    store.configs[0]["model_label_aliases"] = model_aliases
    atomic_json(store.registry_path, store.configs)
    app = create_app(store.home)
    monkeypatch.setattr(training, "has_training_dependencies", lambda: True)
    release = threading.Event()

    def worker(run, stop):
        setup = json.loads((run / "train.json").read_text(encoding="utf-8"))
        assert setup["label_aliases"] == ({"object": "object_1"} if renamed else {})
        assert setup["model_label_aliases"] == model_aliases
        assert setup["classes"][0] == ("object_1" if renamed else "object")
        if cancel:
            assert stop.wait(5)
            raise training.LocalJobCancelled()
        assert release.wait(5)
        (run / training.BEST_WEIGHTS).parent.mkdir(parents=True)
        (run / training.BEST_WEIGHTS).write_bytes(b"checkpoint fixture")
        atomic_json(run / "progress.json", {"epoch": 1, "epochs": 1, "metrics": {"map50": .42}})

    monkeypatch.setattr(app.state.training, "run_worker", worker)
    route = f"/api/projects/{project}/trainings"
    with TestClient(app) as client:
        token = re.search(r'name="labelpilot-token" content="([^"]+)"', client.get("/").text).group(1)
        headers = {"x-labelpilot-token": token}

        def wait_for(expected):
            deadline = time.monotonic() + 7
            while time.monotonic() < deadline:
                result = client.get(route).json()[0]
                if result["status"] in expected:
                    return result
                time.sleep(.02)
            pytest.fail(f"Training did not reach {expected}: {result}")

        assert client.post(route, json={}).status_code == 403
        assert client.post(route, headers=headers, json={"epochs": True}).status_code == 422
        start = client.post(route, headers=headers, json={"epochs": 1, "device": "cpu"})
        assert start.status_code == 202
        job = wait_for({"running"})
        endpoint = f"{route}/{job['id']}"
        manifest = client.get(f"{endpoint}/dataset").json()
        assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 0}
        assert {row["photo_id"] for row in manifest["items"]} == {"a", "b", "c"}
        assert client.post(route, headers=headers, json={}).status_code == 409
        with pytest.raises(Problem, match="本地训练"):
            with app.state.inference.prediction_slot():
                pytest.fail("inference entered the locked training slot")
        if cancel:
            assert client.post(f"{endpoint}/cancel", headers=headers, json={}).status_code == 200
            wait_for({"cancelled"})
            assert client.get(f"{endpoint}/weights").status_code == 404
        else:
            release.set()
            finished = wait_for({"completed"})
            assert finished["progress"]["metrics"]["map50"] == .42
            assert client.get(f"{endpoint}/weights").content == b"checkpoint fixture"
        assert not app.state.inference.lock.locked()
