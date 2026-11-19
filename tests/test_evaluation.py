import re
import threading
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot import evaluation, training
from labelpilot.blindsets import freeze_blindset, get_blindset, verified_manifest
from labelpilot.comparison_details import record_frame, write_details
from labelpilot.evaluation_worker import frozen_frames
from labelpilot.server import create_app
from labelpilot.store import Problem, atomic_json, digest, read_json
from test_workflows import annotate, make_project, rectangle


def test_counts_use_one_to_one_same_class_matching_and_count_negatives():
    names = ("object", "other", "absent")
    counts = {name: dict.fromkeys(("ground_truth", "matched", "missed", "false_positives"), 0) for name in names}
    first, second, third = (0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)
    details = evaluation.count_detections(counts, [("object", box) for box in (first, second, third)], [
        ("object", .4, first), ("object", .95, first), ("object", .8, (25, 25, 35, 35)),
        ("other", .9, second), ("object", .75, (40, 40, 45, 50)), ("out_of_scope", .9, second),
    ], .5)
    assert [(box["label"], box["truth_index"], box["iou"]) for box in details] == [
        ("object", 0, 1), ("other", None, None), ("object", None, None),
        ("object", 2, .5), ("object", None, None),
    ]
    evaluation.count_detections(counts, [], [("other", .9, first)], .5)
    assert counts == {
        "object": {"ground_truth": 3, "matched": 2, "missed": 1, "false_positives": 2},
        "other": {"ground_truth": 0, "matched": 0, "missed": 0, "false_positives": 2},
        "absent": {"ground_truth": 0, "matched": 0, "missed": 0, "false_positives": 0},
    }


def test_freeze_repeats_group_selection_and_pins_annotations(tmp_path):
    store, project, _ = make_project(tmp_path, [
        ("a", "red", "first"), ("b", "blue", "first"), ("c", "white", "second"),
        ("d", "black", "second"), ("pending", "green", "third"),
    ])
    for name in ("a", "b", "c", "d"):
        annotate(store, project, name, "reviewed", [rectangle()])
    annotate(store, project, "pending", "draft", [rectangle()])
    options = {"name": "Held out", "percent": 50, "seed": 123, "unit": "source_group"}
    first = freeze_blindset(store, project, options)
    second = freeze_blindset(store, project, options)
    directory, _ = get_blindset(store, project, first["id"])
    other_dir, _ = get_blindset(store, project, second["id"])
    manifest = verified_manifest(directory, first)
    assert manifest["items"] == verified_manifest(other_dir, second)["items"]
    assert {item["photo_id"] for item in manifest["items"]} in ({"a", "b"}, {"c", "d"})
    assert first["images"] == first["objects"] == 2
    frame_path, row, labels, _ = frozen_frames(directory, first, manifest)[0]
    annotate(store, project, row["photo_id"], "reviewed", [rectangle(label="other")])
    assert labels == frozen_frames(directory, first, manifest)[0][2] == [("object", (10, 15, 65, 70))]
    assert row["sha256"] == digest(frame_path)
    annotation = directory / "dataset" / next(iter(first["annotation_sha256"]))
    document = read_json(annotation)
    document["shapes"][0]["label"] = "other"
    atomic_json(annotation, document)
    with pytest.raises(Problem, match="盲测标注与冻结版本不一致"):
        frozen_frames(directory, first, manifest)


@pytest.mark.parametrize("cancel", [False, True])
def test_workbench_reserves_holdout_and_runs_comparison(tmp_path, monkeypatch, cancel):
    store, project, _ = make_project(tmp_path, [
        ("blind", "red", "held"), ("negative", "black", "held"),
        ("same_source", "green", "held"), ("copied", "red", "copy_group"),
        ("train_a", "white", "first"), ("train_b", "blue", "second"), ("pending", "silver", "second"),
    ])
    manifest_path = store.project(project).config["manifest_path"]
    manifest = read_json(manifest_path)
    for task in manifest["tasks"]:
        task["batch"] = "holdout" if task["package_image"] in ("blind.png", "negative.png") else "train"
    atomic_json(manifest_path, manifest)
    store.rescan(project)
    for photo in ("blind", "same_source", "copied", "train_a", "train_b"):
        annotate(store, project, photo, "reviewed", [rectangle()])
    annotate(store, project, "negative", "reviewed", [])
    app = create_app(store.home)
    monkeypatch.setattr(training, "has_training_dependencies", lambda: True)
    monkeypatch.setattr(evaluation, "has_training_dependencies", lambda: True)
    external = tmp_path / "older.pt"
    external.write_bytes(b"older checkpoint fixture")
    release = threading.Event()

    def train_worker(run, stop):
        path = run / training.BEST_WEIGHTS
        path.parent.mkdir(parents=True)
        path.write_bytes(b"local checkpoint fixture")

    def compare_worker(run, stop, module, directory, copies):
        assert module == "labelpilot.evaluation_worker"
        snapshot = read_json(directory / "snapshot.json")
        manifest = verified_manifest(directory, snapshot)
        assert {item["photo_id"] for item in manifest["items"]} == {"blind", "negative"}
        setup = read_json(run / "compare.json")
        assert setup["models"]["old"]["sha256"] == digest(copies / "old.pt")
        assert setup["models"]["new"]["sha256"] == digest(copies / "new.pt")
        assert setup["models"]["old"]["recorded_overlap"] == {"images": 0, "groups": 0, "folders": 0}
        assert setup["models"]["new"]["recorded_overlap"] is None
        if cancel:
            assert stop.wait(8)
            raise training.LocalJobCancelled()
        assert release.wait(8)
        atomic_json(run / "report.json", {"snapshot": snapshot["id"], "models": setup["models"]})

    monkeypatch.setattr(app.state.training, "run_worker", train_worker)
    monkeypatch.setattr(evaluation, "run_local_worker", compare_worker)
    route = f"/api/projects/{project}"
    with TestClient(app) as client:
        token = re.search(r'name="labelpilot-token" content="([^"]+)"', client.get("/").text)[1]
        headers = {"x-labelpilot-token": token}

        def wait_for(endpoint, statuses):
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                record = client.get(endpoint).json()[0]
                if record["status"] in statuses:
                    return record
                time.sleep(.02)
            pytest.fail(f"Expected {statuses}, got {record}")

        body = {"name": "Holdout A", "percent": 100, "batch": "holdout"}
        assert client.post(f"{route}/blindsets", json=body).status_code == 403
        response = client.post(f"{route}/blindsets", headers=headers, json=body)
        assert response.status_code == 201, response.text
        frozen = response.json()
        assert frozen["images"] == 2 and frozen["objects"] == 1 and frozen["confirmed_empty"] == 1
        assert client.get(f"{route}/blindsets").json() == [frozen]

        response = client.post(f"{route}/trainings", headers=headers, json={"batch": "train"})
        assert response.status_code == 202, response.text
        trained = wait_for(f"{route}/trainings", {"completed", "failed"})
        assert trained["status"] == "completed", trained
        assert trained["blind_out"] == 2
        dataset = client.get(f"{route}/trainings/{trained['id']}/dataset").json()
        assert {item["photo_id"] for item in dataset["items"]} == {"train_a", "train_b"}
        assert client.get(f"{route}/comparison-models").json()[0]["path"] == trained["weights"]

        request = {"set_id": frozen["id"], "old_path": trained["weights"], "new_path": str(external)}
        comparisons = f"{route}/comparisons"
        assert client.post(comparisons, headers=headers, json={**request, "new_path": trained["weights"]}).status_code == 422
        response = client.post(comparisons, headers=headers, json=request)
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        active = wait_for(comparisons, {"running", "failed"})
        assert active["status"] == "running", active
        assert client.post(f"{route}/trainings", headers=headers, json={}).status_code == 409
        assert client.get(f"{comparisons}/{run_id}/report").status_code == 404
        if cancel:
            assert client.post(f"{comparisons}/{run_id}/cancel", headers=headers, json={}).status_code == 200
        else:
            release.set()
        end = wait_for(comparisons, {"completed", "cancelled", "failed"})
        assert end["status"] == ("cancelled" if cancel else "completed"), end
        report = client.get(f"{comparisons}/{run_id}/report")
        if cancel:
            assert report.status_code == 404
        else:
            assert report.status_code == 200 and report.json()["snapshot"] == frozen["id"]
            assert "attachment" in report.headers["content-disposition"]
            assert client.get(f"{comparisons}/{run_id}/frames", params={"label": "object", "kind": "old_missed"}).status_code == 404
        assert client.get(f"/api/projects/other/comparisons/{run_id}/report").status_code == 404
        assert not app.state.inference.lock.locked()


def test_image_review_tracks_targets_even_when_total_misses_are_the_same(tmp_path):
    store, project, source = make_project(tmp_path, [
        ("positive", "red", "held"), ("negative_a", "black", "other"), ("negative_b", "gray", "other"),
    ])
    first, second = (10, 15, 35, 39), (48, 50, 80, 85)
    annotate(store, project, "positive", "reviewed", [
        rectangle(points=[list(box[:2]), list(box[2:])]) for box in (first, second)
    ])
    for photo in ("negative_a", "negative_b"):
        annotate(store, project, photo, "reviewed", [])
    frozen = freeze_blindset(store, project, {"name": "Audit", "percent": 100})
    directory, _ = get_blindset(store, project, frozen["id"])
    frames = frozen_frames(directory, frozen, verified_manifest(directory, frozen))
    positive_image, positive, *_ = next(frame for frame in frames if frame[1]["photo_id"] == "positive")
    run_id = "a" * 32
    run = store.data / "comparisons" / run_id
    run.mkdir(parents=True)
    owner = {"id": run_id, "project": project, "set_id": frozen["id"], "status": "completed"}
    atomic_json(run / "job.json", owner)
    with write_details(run) as detail:
        for role, box in (("old", first), ("new", second)):
            counts = {"object": dict.fromkeys(("ground_truth", "matched", "missed", "false_positives"), 0)}
            for index, (_, item, labels, size) in enumerate(frames):
                raw = [("object", .8, box)] if labels or role == "new" else []
                predictions = evaluation.count_detections(counts, labels, raw, .5)
                record_frame(detail, role, index, item, labels, size, predictions)
            assert counts["object"]["missed"] == 1

    # Change the live annotation and pixels after evaluation. The viewer uses the frozen pair.
    annotate(store, project, "positive", "reviewed", [rectangle(label="other")])
    Image.new("RGB", (96, 96), "blue").save(source / "positive.png")
    url = f"/api/projects/{project}/comparisons/{run_id}/frames"
    with TestClient(create_app(store.home)) as client:
        for kind in ("old_missed", "new_missed", "newly_missed", "fixed_missed"):
            page = client.get(url, params={"label": "object", "kind": kind}).json()
            assert page["total"] == page["count"] == 1
            assert page["items"] == [{"id": positive["id"], "photo_id": "positive", "count": 1}]
        params = {"label": "object", "kind": "new_false_positives", "limit": 1}
        page = client.get(url, params=params).json()
        next_page = client.get(url, params={**params, "offset": 1}).json()
        assert page["total"] == page["count"] == 2
        assert page["items"][0]["photo_id"] == "negative_a" and next_page["items"][0]["photo_id"] == "negative_b"
        assert client.get(url, params={**params, "kind": "anything"}).status_code == 422

        endpoint = f"{url}/{positive['id']}"
        frame = client.get(endpoint).json()
        assert frame["truth"] == [{"label": "object", "box": list(box)} for box in (first, second)]
        assert frame["old"][0]["truth_index"] == 0 and frame["new"][0]["truth_index"] == 1
        response = client.get(f"{endpoint}/image")
        assert response.headers["content-type"] == "image/png"
        with Image.open(BytesIO(response.content)) as image:
            assert image.getpixel((0, 0)) == (255, 0, 0)
        assert client.get(f"{url}/unknown/image").status_code == 404
        assert client.get(f"/api/projects/other/comparisons/{run_id}/frames/{positive['id']}/image").status_code == 404

        positive_image.write_bytes((source / "positive.png").read_bytes())
        assert client.get(f"{endpoint}/image").status_code == 409
