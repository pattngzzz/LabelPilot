import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from labelpilot import evaluation, training
from labelpilot.automation import assess_gates
from labelpilot.blindsets import freeze_blindset
from labelpilot.server import create_app
from labelpilot.store import Store, atomic_json, read_json
from test_efficiency import local_headers
from test_workflows import annotate, make_project, rectangle


def advance(app, project_id, target, *, schedule=False):
    manager = app.state.automation
    deadline = time.monotonic() + 9
    while time.monotonic() < deadline:
        if schedule:
            manager.next_check = 0
        manager.tick()
        runs = manager.view(project_id)["runs"]
        result = next((row for row in runs if target(row)), None)
        if result:
            return result
        time.sleep(.025)
    pytest.fail(f"workflow did not finish: {runs}")


def test_batch_retry_and_watch_are_incremental_with_routing(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    sub = root / "incoming"
    sub.mkdir()
    Image.new("RGB", (96, 96), "#172135").save(root / "base.png")
    for name, color in (("a.png", "#e9ac5c"), ("c.png", "#67acc4")):
        image = Image.new("RGB", (96, 96), "#202930")
        ImageDraw.Draw(image).rectangle((15, 15, 68, 72), fill=color)
        image.save(sub / name)
    (sub / "b-copy.png").write_bytes((sub / "a.png").read_bytes())
    (sub / "broken.png").write_bytes(b"incomplete")
    old, other = tmp_path / "old.pt", tmp_path / "second.pt"
    old.write_bytes(b"primary fixture")
    other.write_bytes(b"second fixture")
    store = Store(tmp_path / "tool")
    project = store.register({"name":"Incoming", "image_root":str(root), "image_files":["base.png"],
                              "model_path":str(old), "classes":["object","other"]})["id"]
    app = create_app(store.home)
    routes = []

    def predict(pid, item_id, options, *, model_path, model_sha256):
        current = app.state.store.detail(pid, item_id)
        photo = current["item"]["photo_id"]
        role = "other" if model_path == str(other) else "main"
        routes.append((photo, role))
        if photo == "c" and role == "main" and routes.count((photo, role)) == 1:
            raise RuntimeError("one temporary model failure")
        label = "other" if photo == "a" and role == "other" else "object"
        return {"base_version":current["version"], "image_sha256":current["image_sha256"],
                "shapes":[rectangle(label=label, score=.94)], "provenance":{"path":model_path}}

    monkeypatch.setattr(app.state.inference, "predict", predict)
    client = TestClient(app)
    headers = local_headers(client)
    url = f"/api/projects/{project}/automation"
    opts = {"intake":{"folder":"incoming", "companion":str(other), "rare":0, "sample":100}}
    assert client.put(url, headers=headers, json=opts).status_code == 200
    assert client.put(url, headers=headers, json={"intake":{"folder":".."}}).status_code == 422
    started = client.post(f"{url}/runs", headers=headers, json={"kind":"intake"}).json()["id"]
    app.state.automation.tick()
    assert app.state.store.project(project).summary()["total"] == 1
    first = advance(app, project, lambda row: row["id"] == started and row["status"] == "failed")
    assert first["phase"] == "predict" and first["duplicates"] == 1
    assert client.post(f"{url}/runs/{started}/retry", headers=headers, json={}).status_code == 202
    ready = advance(app, project, lambda row: row["id"] == started and row["status"] == "partial")
    assert ready["included"] == ready["routed"] == 2 and ready["samples"] == 1
    assert routes.count(("a", "main")) == 1 and routes.count(("c", "main")) == 2
    result = client.get(f"/api/projects/{project}/items", params={"status":"triage", "batch":ready["batch"]}).json()
    assert [row["photo_id"] for row in result["items"]] == ["a","c"]
    assert result["items"][0]["route"]["reasons"] == ["disagree"]
    assert result["items"][1]["route"]["sample"]
    assert all(row["status"] == "predicted" for row in result["items"])
    assert Store(store.home).project(project).summary()["total"] == 3

    Image.new("RGB", (96, 96), "#451762").save(sub / "broken.png")
    assert client.post(f"{url}/runs/{started}/retry", headers=headers, json={}).status_code == 202
    fixed = advance(app, project, lambda row: row["id"] == started and row["status"] == "completed")
    assert fixed["included"] == fixed["routed"] == 3
    assert routes.count(("a", "other")) == 1 and routes.count(("broken", "main")) == 1

    assert client.put(url, headers=headers, json={"intake":{**opts["intake"], "watch":True}}).status_code == 200
    path = sub / "watched.png"
    path.write_bytes(b"image still writing")
    app.state.automation.tick()
    assert len(app.state.automation.view(project)["runs"]) == 1
    Image.new("RGB", (96, 96), "#f1b274").save(path)
    watched = advance(app, project, lambda row: row["id"] != started and row["status"] == "completed", schedule=True)
    assert watched["included"] == 1 and routes.count(("watched", "main")) == 1


def test_automatic_release_excludes_holdout_and_requires_explicit_activation(tmp_path, monkeypatch):
    store, project, _ = make_project(tmp_path, [("train1", "red", "camera-a"), ("train2", "blue", "camera-b"),
                                              ("held", "green", "camera-c")])
    path = store.project(project).config["manifest_path"]
    manifest = read_json(path)
    for row in manifest["tasks"]:
        row["batch"] = "held" if row["package_image"] == "held.png" else "train"
    atomic_json(path, manifest)
    store.rescan(project)
    annotate(store, project, "held", "reviewed", [rectangle()])
    annotate(store, project, "train1", "reviewed", [rectangle()])
    old = tmp_path / "baseline.pt"
    old.write_bytes(b"baseline")
    store.update_model(project, old)
    frozen = freeze_blindset(store, project, {"name":"Independent", "batch":"held", "percent":100})
    app = create_app(store.home)
    monkeypatch.setattr(training, "has_training_dependencies", lambda: True)
    monkeypatch.setattr(evaluation, "has_training_dependencies", lambda: True)

    def train_worker(run, stop):
        weights = run / "run/weights/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"candidate")

    def compare_worker(run, stop, module, directory, inputs):
        setup = read_json(run / "compare.json")
        atomic_json(run / "report.json", {"models":setup["models"], "classes":[
            {"name":"object", "new":{"recall":1.0, "false_positives":0}},
        ]})

    monkeypatch.setattr(app.state.training, "run_worker", train_worker)
    monkeypatch.setattr(evaluation, "run_local_worker", compare_worker)
    client = TestClient(app)
    headers = local_headers(client)
    url = f"/api/projects/{project}/automation"
    settings = {"release":{"enabled":True, "blindset":frozen["id"], "min_images":2, "min_each":1,
                           "gates":{"object":{"recall":.8, "false_positives":0}}, "epochs":1}}
    assert client.put(url, headers=headers, json=settings).status_code == 200
    app.state.automation.tick()
    assert not app.state.automation.view(project)["runs"]
    annotate(store, project, "train2", "reviewed", [rectangle()])
    result = advance(app, project, lambda row: row["status"] in ("passed", "rejected", "failed"), schedule=True)
    assert result["status"] == "passed", result
    assert result["gates"][0]["passed"] and result["reviewed"] == 2
    assert app.state.store.project(project).config["model_path"] == str(old)
    manifest = read_json(app.state.training.artifact(project, result["training_id"], "dataset"))
    assert {row["photo_id"] for row in manifest["items"]} == {"train1","train2"}
    assert client.post(f"{url}/runs/{result['id']}/activate", headers=headers, json={}).status_code == 200
    assert app.state.store.project(project).config["model_path"].endswith("best.pt")
    app.state.automation.next_check = 0
    app.state.automation.tick()
    assert len(app.state.automation.view(project)["runs"]) == 1


def test_missing_truth_and_excess_false_positives_fail_release_gate():
    rules = {"object":{"recall":.8,"false_positives":1}, "other":{"recall":.8, "false_positives":1}}
    report = {"classes":[{"name":"object","new":{"recall":1,"false_positives":2}}]}
    object_result, other_result = assess_gates(report, rules)
    assert not object_result["passed"] and not other_result["passed"]
    assert other_result["recall"] is None


def test_optical_flow_writes_moving_box_and_stops_at_occlusion(tmp_path):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = tmp_path / "frames"
    root.mkdir()
    rng = np.random.default_rng(24)
    patch = Image.fromarray(rng.integers(110, 245, (58, 78), dtype=np.uint8))
    for i in range(4):
        image = Image.new("L", (320, 220), 28)
        if i != 2:
            image.paste(patch, (52 + i*7, 58 + i*4))
        image.save(root / f"t_{i:03d}.png")
    store = Store(tmp_path / "tool")
    project = store.register({"name":"Tracking", "image_root":str(root), "classes":["object"]})["id"]
    first = store.project(project).items[0]
    annotated = annotate(store, project, first["photo_id"], "reviewed", [rectangle(points=[[52,58],[130,116]])])
    current = store.detail(project, first["id"])
    request = {"start":{"id":first["id"],"shape":0,"version":annotated["version"],
                        "image_sha256":current["image_sha256"], "source_fingerprint":current["source_fingerprint"]}, "limit":3}
    client = TestClient(create_app(store.home))
    headers = local_headers(client)
    url = f"/api/projects/{project}/tracking"
    preview = client.post(f"{url}/preview", headers=headers, json=request)
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert result["stop"]["photo_id"] == "t_002"
    target, = result["targets"]
    assert target["box"] == pytest.approx([59,62,137,120], abs=2)
    assert store.detail(project, target["id"])["status"] == "unstarted"
    applied = client.post(f"{url}/apply", headers=headers, json={**request,"token":result["token"]})
    assert applied.status_code == 200, applied.text
    modified = store.detail(project, target["id"])
    assert modified["status"] == "draft"
    assert modified["document"]["shapes"][0]["annotation_origin"] == "optical_flow_track"
    assert store.detail(project, result["stop"]["id"])["status"] == "unstarted"
