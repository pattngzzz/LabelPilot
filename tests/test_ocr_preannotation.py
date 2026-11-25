import time

from PIL import Image

from labelpilot.inference import Inference, Jobs
from labelpilot.ocr_runtime import PaddleOCRRuntime
from labelpilot.store import Store


def make_ocr_project(tmp_path, count=1):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(count):
        Image.new("RGB", (160, 100), "white").save(image_root / f"panel-{index}.png")
    store = Store(tmp_path / "tool")
    project = store.register({"name": "OCR preannotation", "task_type": "ocr",
                              "image_root": str(image_root), "classes": ["text"]})
    return store, project["id"]


def test_ocr_runtime_status_handles_invalid_path_configuration(tmp_path):
    home = tmp_path / "tool"
    data = home / "data"
    data.mkdir(parents=True)
    (data / "ocr_runtime.json").write_text(
        '{"python_executable": null, "model_cache": "D:/cache", '
        '"model_directory": "D:/models", "device": "gpu:0"}',
        encoding="utf-8",
    )

    status = PaddleOCRRuntime(home).status()

    assert status["configured"] is False
    assert status["runtime_available"] is False
    assert "配置路径" in status["message"]


def test_ocr_inference_maps_text_candidates_to_reviewable_shapes(tmp_path, monkeypatch):
    store, project_id = make_ocr_project(tmp_path)
    inference = Inference(store)
    monkeypatch.setattr(inference.ocr_runtime, "predict", lambda *args, **kwargs: (
        [{"points": [[20, 10], [120, 10], [120, 42], [20, 42]], "text": "TEMP 25.0 C", "score": 0.91}],
        {"device": "gpu:0"},
    ))
    item = store.project(project_id).items[0]

    prediction = inference.predict(project_id, item["id"], {"text_det_thresh": 0.35})

    assert prediction["provenance"]["human_reviewed"] is False
    assert prediction["shapes"] == [{
        "label": "text", "shape_type": "polygon",
        "points": [[20.0, 10.0], [120.0, 10.0], [120.0, 42.0], [20.0, 42.0]],
        "text_raw": "TEMP 25.0 C", "readability": "readable", "value_role": "unknown",
        "value_decimal": None, "unit": None, "field_name": None, "channel": None,
        "parent_id": None, "score": 0.91, "text_recognition_confidence": 0.91,
        "prediction_origin": "paddleocr_prediction", "prediction_id": f"{item['id']}:paddleocr:0",
    }]


def test_ocr_batch_saves_predictions_as_pending_review_without_source_edits(tmp_path, monkeypatch):
    store, project_id = make_ocr_project(tmp_path, count=2)
    inference = Inference(store)
    monkeypatch.setattr(inference.ocr_runtime, "predict", lambda *args, **kwargs: (
        [{"points": [[15, 12], [90, 12], [90, 35], [15, 35]], "text": "P 1.25", "score": 0.84}],
        {"device": "gpu:0"},
    ))
    jobs = Jobs(store, inference)
    before = {item["path"]: open(item["path"], "rb").read() for item in store.project(project_id).items}

    job = jobs.start(project_id, {"limit": 2, "text_det_thresh": 0.3})
    deadline = time.monotonic() + 5
    while jobs.lookup(project_id, job["id"])["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.01)

    result = jobs.lookup(project_id, job["id"])
    assert result["status"] == "completed"
    assert result["done"] == 2 and result["failed"] == 0
    for item in store.project(project_id).items:
        current = store.detail(project_id, item["id"])
        assert current["status"] == "predicted"
        assert current["document"]["shapes"][0]["text_raw"] == "P 1.25"
        assert current["document"]["shapes"][0]["prediction_origin"] == "paddleocr_prediction"
        assert open(item["path"], "rb").read() == before[item["path"]]
