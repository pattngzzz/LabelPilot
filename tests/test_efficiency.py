import re
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from labelpilot.annotation_tools import segment_prompt
from labelpilot.server import create_app
from labelpilot.store import atomic_json, read_json
from test_workflows import annotate, make_project, rectangle


def local_headers(client):
    return {"x-labelpilot-token": re.search(r'name="labelpilot-token" content="([^"]+)"', client.get("/").text)[1]}


def test_keyframes_follow_explicit_order_and_write_only_checked_drafts(tmp_path):
    store, project, _ = make_project(tmp_path, [
        ("start", "red", "take"), ("second", "green", "take"), ("accepted", "blue", "take"),
        ("middle", "white", "take"), ("end", "black", "take"), ("outside", "gray", "take"),
        ("other_take", "silver", "different"),
    ])
    path = store.project(project).config["manifest_path"]
    manifest = read_json(path)
    for index, task in enumerate(manifest["tasks"]):
        task["frame_index"] = index * 10
    atomic_json(path, manifest)
    store.rescan(project)
    annotate(store, project, "start", "reviewed", [rectangle(points=[[10, 10], [30, 30]], score=.8)])
    annotate(store, project, "end", "reviewed", [rectangle(points=[[50, 30], [70, 50]])])
    annotate(store, project, "accepted", "reviewed", [])
    ids = {item["photo_id"]: item["id"] for item in store.project(project).items}

    def anchor(photo):
        current = store.detail(project, ids[photo])
        return {"id": ids[photo], "shape": 0, **{key: current[key] for key in ("version", "image_sha256", "source_fingerprint")}}

    with TestClient(create_app(store.home)) as client:
        headers = local_headers(client)
        route = f"/api/projects/{project}/sequence"
        result = client.post(f"{route}/inherit", headers=headers, json=anchor("start")).json()
        assert result == {"id": ids["second"], "appended": True}
        repeat = client.post(f"{route}/inherit", headers=headers, json=anchor("start")).json()
        assert repeat == {"id": ids["second"], "appended": False}
        copied = store.detail(project, ids["second"])
        assert copied["status"] == "draft"
        assert copied["document"]["shapes"][0]["annotation_origin"] == "inherit_previous"
        assert "score" not in copied["document"]["shapes"][0]

        request = {"start": anchor("start"), "end": anchor("end")}
        preview = client.post(f"{route}/preview", headers=headers, json=request).json()
        assert [row["photo_id"] for row in preview["targets"]] == ["middle"]
        assert preview["targets"][0]["box"] == pytest.approx([40, 25, 60, 45])
        assert preview["skipped"] == {"status": 1, "matched": 1, "unreadable": 0}
        assert store.detail(project, ids["middle"])["status"] == "unstarted"
        annotate(store, project, "middle", "draft", [rectangle(label="other")])
        assert client.post(f"{route}/apply", headers=headers, json={**request, "token": preview["token"]}).status_code == 409
        preview = client.post(f"{route}/preview", headers=headers, json=request).json()
        assert client.post(f"{route}/apply", headers=headers, json={**request, "token": preview["token"]}).status_code == 200
        target = store.detail(project, ids["middle"])
        assert target["status"] == "draft"
        assert [shape["label"] for shape in target["document"]["shapes"]] == ["other", "object"]
        assert target["document"]["shapes"][1]["end_item_id"] == ids["end"]
        assert store.detail(project, ids["accepted"])["status"] == "reviewed"
        assert store.detail(project, ids["outside"])["status"] == "unstarted"


def test_near_duplicate_queue_keeps_distinct_images_and_originals(tmp_path):
    store, project, paths = make_project(tmp_path, [
        ("original", "white", "first"), ("jpeg_copy", "black", "first"),
        ("different", "red", "second"), ("already_reviewed", "white", "second"),
    ])
    image = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(image).rectangle((23, 19, 67, 70), fill="teal")
    image.save(paths / "original.png")
    image.save(paths / "jpeg_copy.png", "JPEG", quality=77)
    annotate(store, project, "already_reviewed", "reviewed", [])
    original = {path.name: path.read_bytes() for path in paths.glob("*.png")}
    route = f"/api/projects/{project}"
    with TestClient(create_app(store.home)) as client:
        headers = local_headers(client)
        candidates = client.post(f"{route}/similarity/scan", headers=headers, json={"batch": "images"}).json()
        assert candidates["images"] == 3
        group, = candidates["groups"]
        assert {image["photo_id"] for image in group["members"]} == {"original", "jpeg_copy"}
        duplicate = next(image["id"] for image in group["members"] if image["photo_id"] == "jpeg_copy")
        assert client.post(f"{route}/similarity/selection", headers=headers,
                           json={"batch":"images", "excluded":[duplicate]}).status_code == 200
        kept = client.get(f"{route}/items", params={"status":"curated", "batch":"images"}).json()
        assert {item["photo_id"] for item in kept["items"]} == {"original", "different"}
        assert client.get(f"{route}/items").json()["total"] == 4
    assert original == {path.name: path.read_bytes() for path in paths.glob("*.png")}


def test_prompt_assistance_follows_positive_component():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (160, 100), "#13202f")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 22, 78, 69), fill="#edc13b")
    draw.rectangle((99, 22, 146, 69), fill="#edc13b")
    assert segment_prompt(image, [[52, 48]], [[103, 45], [5, 4]], image.size) == pytest.approx([30, 22, 79, 70])


def test_suspicion_job_records_three_reasons_without_editing_annotations(tmp_path, monkeypatch):
    store, project, _ = make_project(tmp_path, [("manual", "blue", "first"), ("pending", "black", "second")])
    annotate(store, project, "manual", "reviewed", [
        rectangle(points=[[10,10],[25,25]]), rectangle(label="other", points=[[40,35],[60,57]])
    ])
    item = next(item["id"] for item in store.project(project).items if item["photo_id"] == "manual")
    before = store.detail(project, item)
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"fixture model; no deserialization")
    store.update_model(project, model)
    app = create_app(store.home)

    def predict(project_id, item_id, options):
        assert item_id == item
        predictions = [
            rectangle(label="other", points=[[10,10],[25,25]], score=.95),
            rectangle(label="other", points=[[49,42],[74,65]], score=.92),
            rectangle(points=[[70,12],[93,30]], score=.89),
        ]
        return {"base_version": before["version"], "image_sha256": before["image_sha256"], "shapes": predictions,
                "provenance": {"model_path": str(model), "confidence": options["confidence"]}}

    monkeypatch.setattr(app.state.inference, "predict", predict)
    route = f"/api/projects/{project}"
    with TestClient(app) as client:
        headers = local_headers(client)
        response = client.post(f"{route}/audits", headers=headers, json={"confidence": .6})
        assert response.status_code == 202
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"{route}/jobs").json()[0]
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(.02)
        assert (job["status"], job["flagged"], job["total"]) == ("completed", 1, 1), job
        result = client.get(f"{route}/items", params={"status":"audit"}).json()
        summary = result["items"][0]["audit"]
        assert summary["kinds"] == {"missing": 1, "class_conflict": 1, "box_offset": 1}
        endpoint = f"{route}/items/{item}/audit"
        evidence = client.get(endpoint).json()
        assert len(evidence["findings"]) == 3
        assert client.get(f"{route}/items/{item}").json()["document"] == before["document"]
        assert client.post(f"{endpoint}/dismiss", headers=headers, json={"id":"older-evidence"}).status_code == 409
        assert client.post(f"{endpoint}/dismiss", headers=headers, json={"id":evidence["id"]}).status_code == 200
        assert client.get(endpoint).json() is None
        assert store.detail(project, item)["version"] == before["version"]
