import copy
import json
import threading
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot.duplicates import duplicate_threshold, filter_duplicates
from labelpilot.exporting import export_project, preview_export
from labelpilot.inference import Inference, Jobs, validate_model_task
from labelpilot.sam3_worker import OnnxRuntime, validate_targets
from labelpilot.server import create_app
from labelpilot.store import Problem, Store
from labelpilot.suggestions import AssistanceJobs, load_contours, predict_contours, save_contours


def rectangle(label="part", left=10, top=10, right=40, bottom=40, score=.9, **extra):
    return {"label": label, "shape_type": "rectangle", "points": [[left, top], [right, bottom]], "score": score, **extra}


@pytest.fixture
def scene(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 80), "#314259").save(images / "frame.png")
    store = Store(tmp_path / "workspace")
    project = store.project(store.register({"name": "test", "task_type": "detect", "image_root": str(images),
                                          "classes": ["part", "panel"]})["id"])
    return store, project, project.items[0]["id"]


def save_shapes(scene, shapes, status="draft"):
    store, project, item_id = scene
    current = store.detail(project.id, item_id)
    document = copy.deepcopy(current["document"])
    document["shapes"] = shapes
    return store.save(project.id, item_id, {**current, "document": document, "expected_version": current["version"],
                                          "status": status}, predicted=status == "predicted")


class FakeSam3:
    slot_reserved = False

    def __init__(self):
        self.calls = []

    def config(self):
        return {"backend": "onnx"}

    def predict(self, path, payload, **kwargs):
        self.calls.append((payload, kwargs))
        candidates = []
        for target in payload["targets"]:
            left, top, right, bottom = target["box"]
            candidates.append({"source_index": target["source_index"], "score": .95,
                               "points": [[left, top], [right, top], [right, bottom], [left, bottom]]})
        return {"width": 100, "height": 80, "candidates": candidates}


def wait_job(jobs, job):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        current = jobs.lookup(job["project"], job["id"])
        if current["status"] not in ("queued", "running"):
            return current
        time.sleep(.01)
    pytest.fail("background assistance job did not finish")


def test_duplicates_preserve_multiple_instances_and_different_classes():
    first = rectangle(score=.7)
    duplicate = rectangle(left=11, top=11, score=.95)
    distinct = rectangle(left=60, right=90)
    other_class = rectangle("panel")
    shapes, removed = filter_duplicates([first, duplicate, distinct, other_class], .65)
    assert shapes == [duplicate, distinct, other_class] and removed == [0]


def test_detection_weights_can_prelabel_segmentation_but_not_train_it():
    model = SimpleNamespace(task="detect")
    validate_model_task(model, "segment", allow_detection=True)
    with pytest.raises(Problem, match="项目需要"):
        validate_model_task(model, "segment")
    with pytest.raises(Problem, match="项目需要"):
        validate_model_task(model, "semantic", allow_detection=True)


def test_duplicates_never_remove_protected_human_objects_or_polygons():
    human = rectangle(score=None)
    polygon = {**rectangle(), "shape_type": "polygon", "points": [[10, 10], [40, 10], [40, 40]]}
    kept, removed = filter_duplicates([rectangle(), polygon], .65, protected=[human])
    assert kept == [polygon] and removed == [0]
    assert human["score"] is None


@pytest.mark.parametrize("value", [True, None, .4, 1.1, "0.8", float("nan")])
def test_duplicate_threshold_rejects_invalid_values(value):
    with pytest.raises(Problem):
        duplicate_threshold({"duplicate_iou": value})


def test_contours_are_linked_to_boxes_and_do_not_change_annotations(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle(), rectangle("panel", left=50, right=90)])
    before = store.detail(project.id, item_id)
    assistant = FakeSam3()
    result = predict_contours(store, assistant, project.id, item_id, {})
    save_contours(store, project.id, item_id, result)
    assert store.detail(project.id, item_id) == before
    assert load_contours(store, project.id, item_id) == result
    assert [candidate["label"] for candidate in result["candidates"]] == ["part", "panel"]
    assert result["candidates"][0]["source_shape"] == before["document"]["shapes"][0]
    assert assistant.calls[0][0]["targets"][1]["prompt"] == "panel"
    save_shapes(scene, [rectangle(left=12)])
    assert load_contours(store, project.id, item_id) == {"candidates": [], "stale": True}


def test_contours_reject_changed_sources_and_bad_geometry(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle()])
    assistant = FakeSam3()
    original = assistant.predict

    def changed(path, payload):
        result = original(path, payload)
        save_shapes(scene, [rectangle(left=12)])
        return result

    assistant.predict = changed
    with pytest.raises(Problem, match="推理期间"):
        predict_contours(store, assistant, project.id, item_id, {})
    assistant.predict = lambda *args, **kwargs: {"width": 100, "height": 80, "candidates": [
        {"source_index": 0, "score": .9, "points": [[-1, 0], [40, 0], [40, 40]]}]}
    with pytest.raises(Problem, match="边界"):
        predict_contours(store, assistant, project.id, item_id, {})


def test_contours_choose_one_matching_candidate_per_source_box(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle()])
    assistant = FakeSam3()
    assistant.predict = lambda *args: {"width": 100, "height": 80, "candidates": [
        {"source_index": 0, "score": .99, "points": [[60, 50], [90, 50], [90, 70]]},
        {"source_index": 0, "score": .9, "points": [[10, 10], [40, 10], [40, 40]]},
        {"source_index": 0, "score": .8, "points": [[10, 10], [40, 10], [40, 40]]}]}
    result = predict_contours(store, assistant, project.id, item_id, {})
    assert len(result["candidates"]) == 1 and result["candidates"][0]["score"] == .9


def test_batch_contours_persist_separate_reviewable_suggestions(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle()], "reviewed")
    before = store.detail(project.id, item_id)
    jobs = Jobs(store, Inference(store))
    assistant = FakeSam3()
    service = AssistanceJobs(store, jobs, assistant)
    job = wait_job(jobs, service.start(project.id, {"limit": 10}))
    assert job["status"] == "completed" and job["candidates"] == 1
    assert store.detail(project.id, item_id) == before
    assert load_contours(store, project.id, item_id)["candidates"][0]["source_index"] == 0
    assert assistant.calls[0][1]["owner_job_id"] == job["id"]
    with pytest.raises(Problem, match="已有候选"):
        service.start(project.id, {"limit": 10})
    rerun = wait_job(jobs, service.start(project.id, {"limit": 10, "regenerate": True}))
    assert rerun["status"] == "completed" and rerun["candidates"] == 1


def test_cancelling_contour_batch_does_not_save_inflight_results(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle()])
    before = store.detail(project.id, item_id)
    entered, finish = threading.Event(), threading.Event()
    assistant = FakeSam3()
    original = assistant.predict

    def paused(path, payload, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(path, payload, **kwargs)

    assistant.predict = paused
    jobs = Jobs(store, Inference(store))
    service = AssistanceJobs(store, jobs, assistant)
    job = service.start(project.id, {})
    assert entered.wait(3)
    jobs.cancel(job["id"])
    finish.set()
    completed = wait_job(jobs, job)
    assert completed["status"] == "cancelled" and completed["done"] == 0
    assert load_contours(store, project.id, item_id)["candidates"] == []
    assert store.detail(project.id, item_id) == before


def test_batch_limit_ignores_polygons_and_previously_completed_candidates(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle()])
    root = project.root
    Image.new("RGB", (100, 80), "#335577").save(root / "second.png")
    store.rescan(project.id)
    project = store.project(project.id)
    second = next(item["id"] for item in project.items if item["id"] != item_id)
    save_shapes((store, project, second), [rectangle("panel")])
    assistant = FakeSam3()
    jobs = Jobs(store, Inference(store))
    service = AssistanceJobs(store, jobs, assistant)
    first = wait_job(jobs, service.start(project.id, {"limit": 1}))
    other = wait_job(jobs, service.start(project.id, {"limit": 1}))
    assert first["done"] == other["done"] == 1
    assert load_contours(store, project.id, item_id)["candidates"]
    assert load_contours(store, project.id, second)["candidates"]


def test_model_slot_owner_is_only_exempt_for_its_own_batch(scene):
    store, project, _ = scene
    inference = Inference(store)
    jobs = Jobs(store, inference)
    jobs.records["own"] = {"id": "own", "status": "running"}
    jobs.reserve_model_slot(owner_job_id="own")
    assert inference.exclusive.is_set()
    jobs.release_model_slot()
    jobs.records["other"] = {"id": "other", "status": "queued"}
    with pytest.raises(Problem, match="等待"):
        jobs.reserve_model_slot(owner_job_id="own")


def test_batch_cleanup_keeps_human_shapes_and_every_history_revision(scene):
    store, project, item_id = scene
    model = rectangle(prediction_origin="model_prediction")
    duplicate = rectangle(left=11, score=.8, prediction_origin="model_prediction")
    human = rectangle("panel", score=None)
    save_shapes(scene, [model, duplicate, human], "predicted")
    jobs = Jobs(store, Inference(store))
    job = wait_job(jobs, AssistanceJobs(store, jobs, FakeSam3()).start(project.id, {}, cleanup=True))
    assert job["status"] == "completed" and job["removed"] == 1
    current = store.detail(project.id, item_id)
    assert current["status"] == "draft" and current["document"]["shapes"] == [model, human]
    assert len(store.history(project.id, item_id)) == 3
    assert store.history(project.id, item_id)[1]["status"] == "predicted"


def test_cleanup_excludes_reviewed_data_and_protects_unmarked_objects(scene):
    store, project, item_id = scene
    save_shapes(scene, [rectangle(), rectangle(left=11)], "reviewed")
    jobs = Jobs(store, Inference(store))
    with pytest.raises(Problem, match="没有模型待审核"):
        AssistanceJobs(store, jobs, FakeSam3()).start(project.id, {}, cleanup=True)
    assert len(store.detail(project.id, item_id)["document"]["shapes"]) == 2


def test_instance_and_detection_exports_share_one_polygon_annotation(scene):
    store, project, item_id = scene
    polygon = {**rectangle(), "shape_type": "polygon", "points": [[10, 10], [40, 10], [40, 40]]}
    save_shapes(scene, [polygon], "reviewed")
    before = store.detail(project.id, item_id)
    tokens = []
    for task, expected_fields in (("detect", 5), ("segment", 7)):
        options = {"format": "yolo", "output_task": task}
        preview = preview_export(store, project.id, options)
        tokens.append(preview["preview_token"])
        assert preview["blocking_issue"] is None
        exported = export_project(store, project.id, options)
        with zipfile.ZipFile(store.home / "exports" / exported["filename"]) as archive:
            labels = next(name for name in archive.namelist() if name.startswith("labels/"))
            assert len(archive.read(labels).decode().split()) == expected_fields
            manifest = json.loads(archive.read("export_manifest.json"))
            assert manifest["task_type"] == task
    assert tokens[0] != tokens[1]
    assert store.detail(project.id, item_id) == before


def test_segmentation_export_refuses_to_drop_unfinished_boxes(scene):
    store, project, _ = scene
    save_shapes(scene, [rectangle()], "reviewed")
    options = {"format": "yolo", "output_task": "segment"}
    assert "未生成轮廓" in preview_export(store, project.id, options)["blocking_issue"]
    with pytest.raises(Problem, match="未生成轮廓"):
        export_project(store, project.id, options)


def test_onnx_box_prompts_encode_image_once_and_preserve_original_coordinates():
    mask = np.zeros((1, 80, 100), dtype=np.uint8)
    mask[:, 10:40, 10:40] = 1
    calls = []
    model = SimpleNamespace(encode_image=lambda path: ({"encoded": True}, 100, 80),
                            predict=lambda path, prompt, threshold, **kwargs: (calls.append(kwargs) or (mask, [.9], 100, 80)))
    runtime = OnnxRuntime.__new__(OnnxRuntime)
    runtime.model = model
    targets = [{"source_index": index, "box": [10, 10, 40, 40], "prompt": "part"} for index in range(2)]
    result = runtime.predict_boxes("image.png", targets, .35)
    assert len(result["candidates"]) == 2
    assert calls[0]["encoded"] is calls[1]["encoded"]
    assert calls[0]["box"] == [10, 10, 40, 40]
    assert min(point[0] for point in result["candidates"][0]["points"]) == 10
    with pytest.raises(ValueError, match="坐标"):
        validate_targets([{**targets[0], "box": [-1, 10, 40, 40]}], 100, 80)


def test_contours_api_round_trip_in_detection_project(tmp_path, monkeypatch):
    app = create_app(tmp_path / "app")
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 80)).save(images / "frame.png")
    store = app.state.store
    project = store.project(store.register({"name": "mixed", "image_root": str(images), "classes": ["part"]})["id"])
    item_id = project.items[0]["id"]
    save_shapes((store, project, item_id), [rectangle()])
    monkeypatch.setattr(app.state.sam3, "predict", FakeSam3().predict)
    headers = {"x-labelpilot-token": app.state.token}
    with TestClient(app) as client:
        url = f"/api/projects/{project.id}/items/{item_id}/contours"
        response = client.post(url, json={}, headers=headers)
        assert response.status_code == 200 and len(response.json()["candidates"]) == 1
        assert client.get(url).json() == response.json()
    assert store.detail(project.id, item_id)["document"]["shapes"][0]["shape_type"] == "rectangle"
