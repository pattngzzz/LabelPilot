import hashlib
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot import native_picker
from labelpilot.inference import Inference, Jobs
from labelpilot.ocr_runtime import PaddleOCRRuntime
from labelpilot.paddleocr_models import inspect_model, model_signature
from labelpilot.server import create_app
from labelpilot.store import Problem, Store


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


def make_runtime(tmp_path):
    home = tmp_path / "tool"
    runtime = PaddleOCRRuntime(home)
    runtime.config_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = {"python_executable": sys.executable, "model_cache": str(tmp_path / "cache"),
                     "model_directory": str(tmp_path / "models"), "device": "cpu", "extra": "keep"}
    runtime.config_path.write_text(json.dumps(configuration), encoding="utf-8")
    return runtime


def make_model(tmp_path, name="PP-OCRv5_server_rec", graph="inference.json"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "inference.yml").write_text(f"Global:\n  model_name: {name}\n", encoding="utf-8")
    (directory / graph).write_bytes(b"graph")
    (directory / "inference.pdiparams").write_bytes(b"weights")
    return directory


def test_custom_ocr_models_persist_inspect_roles_and_restore_defaults(tmp_path):
    runtime = make_runtime(tmp_path)
    detection = make_model(tmp_path, "PP-OCRv5_server_det")
    recognition = make_model(tmp_path, graph="inference.pdmodel")

    status = runtime.configure({"text_detection_model_dir": str(detection),
                                "text_recognition_model_dir": str(recognition)})
    assert status["models_downloaded"] and status["configured"]
    assert status["text_detection_model_dir"] == str(detection)
    assert status["text_recognition_model_dir"] == str(recognition)
    assert status["recognition_model"] == recognition.name
    assert json.loads(runtime.config_path.read_text())["extra"] == "keep"
    assert runtime.set_device("gpu:0")["text_recognition_model_dir"] == str(recognition)

    status = runtime.configure({"text_detection_model_dir": None})
    assert status["text_detection_model_dir"] is None
    assert status["text_recognition_model_dir"] == str(recognition)
    status = runtime.configure({"text_recognition_model_dir": ""})
    assert status["text_recognition_model_dir"] is None
    assert "text_recognition_model_name" not in json.loads(runtime.config_path.read_text())


@pytest.mark.parametrize("payload", [
    {}, {"unknown": "model"}, {"device": "cuda"}, {"text_recognition_model_dir": 8},
    {"text_recognition_model_dir": "relative/path"}, {"text_recognition_model_dir": "bad\x00path"},
])
def test_invalid_model_update_keeps_configuration_and_worker(tmp_path, monkeypatch, payload):
    runtime = make_runtime(tmp_path)
    before = runtime.config_path.read_bytes()
    stop = Mock()
    monkeypatch.setattr(runtime, "_stop_worker", stop)
    with pytest.raises(Problem):
        runtime.configure(payload)
    assert runtime.config_path.read_bytes() == before
    stop.assert_not_called()


def test_mismatched_and_incomplete_models_do_not_replace_saved_configuration(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    recognition = make_model(tmp_path)
    before = runtime.config_path.read_bytes()
    stop = Mock()
    monkeypatch.setattr(runtime, "_stop_worker", stop)
    with pytest.raises(Problem, match="不能混用"):
        runtime.configure({"text_detection_model_dir": str(recognition)})
    (recognition / "inference.pdiparams").unlink()
    with pytest.raises(Problem, match="推理目录"):
        runtime.configure({"text_recognition_model_dir": str(recognition)})
    assert runtime.config_path.read_bytes() == before
    stop.assert_not_called()


def test_missing_old_model_can_be_replaced_or_reset(tmp_path):
    runtime = make_runtime(tmp_path)
    value = json.loads(runtime.config_path.read_text())
    value["text_recognition_model_dir"] = str(tmp_path / "deleted-model")
    runtime.config_path.write_text(json.dumps(value))
    assert not runtime.status()["configured"]
    recognition = make_model(tmp_path)
    assert runtime.configure({"text_recognition_model_dir": str(recognition)})["configured"]
    (recognition / "inference.json").unlink()
    assert runtime.configure({"text_recognition_model_dir": None})["configured"]


def test_model_identity_hashes_yaml_and_weight_changes(tmp_path):
    recognition = make_model(tmp_path, "ch_PP-OCRv4_rec_server")
    before = model_signature(str(recognition))
    inspected = inspect_model(str(recognition), "recognition")
    assert inspected["sha256"]["inference.pdiparams"] == hashlib.sha256(b"weights").hexdigest()
    (recognition / "inference.pdiparams").write_bytes(b"updated-weights")
    assert model_signature(str(recognition)) != before
    assert inspect_model(str(recognition), "recognition")["sha256"] != inspected["sha256"]
    (recognition / "inference.yml").write_text("!!python/object/apply:os.system ['echo unsafe']")
    with pytest.raises(ValueError, match="Global.model_name"):
        inspect_model(str(recognition), "recognition")


def test_worker_stages_custom_models_without_changing_sources(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    recognition = make_model(tmp_path / "中文路径")
    originals = {file.name: file.read_bytes() for file in recognition.iterdir()}
    runtime.configure({"text_recognition_model_dir": str(recognition)})
    monkeypatch.setattr(runtime, "_inspect_models", lambda python, models:
                        {role: inspect_model(path, role) for role, path in models.items()})
    process = Mock(stdin=io.StringIO(), stdout=io.StringIO())
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr("labelpilot.ocr_runtime.subprocess.Popen", launch)
    runtime._start_worker(*runtime._configuration())
    parameters = json.loads(launch.call_args.kwargs["env"]["LABELPILOT_OCR_MODELS"])
    spec = parameters["recognition"]
    assert spec["source_directory"] == str(recognition)
    staged = Path(spec["directory"])
    assert staged.parent.name == "custom_models"
    assert inspect_model(str(staged), "recognition")["sha256"] == spec["sha256"]
    assert {file.name: file.read_bytes() for file in recognition.iterdir()} == originals
    assert runtime.signature == runtime._signature(*runtime._configuration())
    runtime.close()


def test_model_change_during_preparation_blocks_worker_start(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    recognition = make_model(tmp_path)
    runtime.configure({"text_recognition_model_dir": str(recognition)})

    def inspect_then_change(python, models):
        spec = inspect_model(str(recognition), "recognition")
        (recognition / "inference.yml").write_text("Global:\n  model_name: PP-OCRv5_mobile_rec\n")
        return {"recognition": spec}

    monkeypatch.setattr(runtime, "_inspect_models", inspect_then_change)
    launch = Mock()
    monkeypatch.setattr("labelpilot.ocr_runtime.subprocess.Popen", launch)
    with pytest.raises(Problem, match="权重发生变化"):
        runtime._start_worker(*runtime._configuration())
    launch.assert_not_called()
    assert runtime.process is None


@pytest.mark.parametrize("mode,choose_file", [("ocr_model_folder", False), ("ocr_model_file", True)])
def test_native_ocr_picker_normalizes_directory_and_handles_cancel(tmp_path, monkeypatch, mode, choose_file):
    recognition = make_model(tmp_path)
    selected = [str(recognition / "inference.pdiparams" if choose_file else recognition)]
    calls = []

    def run(command, **options):
        calls.append(json.loads(options["input"]))
        return native_picker.subprocess.CompletedProcess(command, 0, json.dumps(selected), "")

    monkeypatch.setattr(native_picker.subprocess, "run", run)
    assert native_picker.select_images(mode) == {"cancelled": False, "path": str(recognition)}
    assert calls[0]["mode"] == mode
    selected.clear()
    assert native_picker.select_images(mode) == {"cancelled": True}
    selected.append(str(tmp_path / "best.pt"))
    with pytest.raises(Problem):
        native_picker.select_images(mode)
    assert not native_picker.PICKER_LOCK.locked()


def test_ocr_model_routes_require_session_and_idle_runtime(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    app = create_app(runtime.home)
    recognition = make_model(tmp_path)
    pick = Mock(return_value={"cancelled": False, "path": str(recognition)})
    monkeypatch.setattr(native_picker, "select_images", pick)
    with TestClient(app) as client:
        headers = {"X-LabelPilot-Token": app.state.token}
        payload = {"text_recognition_model_dir": str(recognition)}
        assert client.post("/api/ocr/runtime", json=payload).status_code == 403
        assert client.post("/api/ocr/models/select", json={"mode": "folder"}).status_code == 403
        assert client.post("/api/ocr/models/select", json={"mode": "folder"},
                           headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        pick.assert_not_called()
        assert client.post("/api/ocr/models/select", json={"mode": "file"}, headers=headers).json()["path"] == str(recognition)
        pick.assert_called_once_with("ocr_model_file", None)
        assert client.post("/api/ocr/models/select", json={"mode": "bad"}, headers=headers).status_code == 422
        assert client.post("/api/ocr/runtime", json=payload, headers=headers).status_code == 200
        before = runtime.config_path.read_bytes()
        for busy in ("job", "lock", "exclusive"):
            if busy == "job":
                app.state.jobs.records["test-job"] = {"status": "running"}
            elif busy == "lock":
                app.state.inference.lock.acquire()
            else:
                app.state.inference.exclusive.set()
            try:
                assert client.post("/api/ocr/runtime", json={"text_recognition_model_dir": None},
                                   headers=headers).status_code == 409
                assert runtime.config_path.read_bytes() == before
            finally:
                app.state.jobs.records.clear()
                if busy == "lock":
                    app.state.inference.lock.release()
                app.state.inference.exclusive.clear()


def test_custom_model_identity_is_recorded_in_reviewable_prediction(tmp_path, monkeypatch):
    store, project_id = make_ocr_project(tmp_path)
    inference = Inference(store)
    identity = {"name": "PP-OCRv5_mobile_rec", "directory": "D:/custom/rec", "sha256": {"inference.pdiparams": "abc"}}
    monkeypatch.setattr(inference.ocr_runtime, "predict", lambda *args, **kwargs: (
        [], {"device": "cpu", "detection_model": "PP-OCRv5_mobile_det",
             "recognition_model": identity["name"], "models": {"recognition": identity}},
    ))
    item = store.project(project_id).items[0]
    prediction = inference.predict(project_id, item["id"], {})
    assert prediction["provenance"]["recognition_model"] == identity["name"]
    assert prediction["provenance"]["models"]["recognition"] == identity
    assert prediction["provenance"]["human_reviewed"] is False
