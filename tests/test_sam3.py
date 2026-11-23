import numpy as np
import pytest
import sys
from types import SimpleNamespace
from PIL import Image
from fastapi.testclient import TestClient

from labelpilot import native_picker, sam3_onnx
from labelpilot.sam3 import Sam3Assistant
from labelpilot.sam3_worker import mask_candidates, mask_polygon
from labelpilot.server import create_app
from labelpilot.store import Problem, Store


def test_sam3_is_idle_until_a_local_checkpoint_is_configured(tmp_path):
    assistant = Sam3Assistant(Store(tmp_path), object(), object())

    status = assistant.status()

    assert status["configured"] is False
    assert status["installed"] is False
    assert status["running"] is False
    assert "本地权重" in status["message"]


def test_sam3_configuration_rejects_missing_paths(tmp_path):
    assistant = Sam3Assistant(Store(tmp_path), object(), object())

    with pytest.raises(Problem, match="有效的 SAM3 Python 环境"):
        assistant.configure({"python_executable": "missing-python.exe", "checkpoint_path": "missing.pt"})


def test_sam3_mask_polygon_uses_source_image_coordinates():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[8:30, 12:39] = 1

    polygon = mask_polygon(mask)

    assert polygon is not None
    assert len(polygon) >= 4
    assert min(point[0] for point in polygon) == 12
    assert max(point[0] for point in polygon) == 38
    assert min(point[1] for point in polygon) == 8
    assert max(point[1] for point in polygon) == 29


def test_sam3_status_and_ui_are_available_before_weights_are_configured(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        status = client.get("/api/sam3")
        page = client.get("/")
        script = client.get("/static/sam3.js")

    assert status.status_code == 200
    assert status.json()["configured"] is False
    assert page.status_code == 200 and 'id="sam3Dialog"' in page.text
    assert script.status_code == 200 and 'itemUrl("/sam3")' in script.text


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch):
    for name in (*sam3_onnx.MODEL_FILES, "tokenizer.json"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "weights.data").write_bytes(b"12345678")
    tensor = SimpleNamespace(data_location=1, dims=[2], data_type=1, external_data=[
        SimpleNamespace(key=key, value=value) for key, value in
        (("location", "weights.data"), ("offset", "0"), ("length", "8"))])
    model = SimpleNamespace(graph=SimpleNamespace(initializer=[tensor], node=[]))
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(
        load_model=lambda *args, **kwargs: model, TensorProto=SimpleNamespace(EXTERNAL=1),
        helper=SimpleNamespace(tensor_dtype_to_np_dtype=lambda value: np.dtype("float32"))))
    return tmp_path, tensor


def test_onnx_bundle_validates_all_external_tensor_ranges(fake_bundle):
    root, _ = fake_bundle
    assert sam3_onnx.validate_bundle(root) == root


@pytest.mark.parametrize("issue", ["truncated", "missing", "traversal"])
def test_onnx_bundle_rejects_incomplete_or_outside_external_data(fake_bundle, issue):
    root, tensor = fake_bundle
    if issue == "truncated":
        (root / "weights.data").write_bytes(b"1234")
    elif issue == "missing":
        (root / "weights.data").unlink()
    else:
        tensor.external_data[0].value = "../outside.data"
    with pytest.raises(ValueError, match="不完整|缺失|越界"):
        sam3_onnx.validate_bundle(root)


def test_onnx_preprocessing_keeps_rgb_channel_order_and_source_dimensions():
    image = Image.new("RGB", (50, 40), (255, 64, 0))
    pixels = sam3_onnx.prepare_image(image, [3, 16, 24], "tensor(uint8)")
    assert image.size == (50, 40)
    assert pixels.shape == (3, 16, 24) and pixels.dtype == np.uint8
    assert pixels[:, 0, 0].tolist() == [255, 64, 0]
    normalized = sam3_onnx.prepare_image(image, [1, 3, 16, 24], "tensor(float)")
    assert normalized.shape == (1, 3, 16, 24)
    assert normalized[0, :, 0, 0].tolist() == pytest.approx([1, 64 / 255 * 2 - 1, -1])


def test_onnx_tokens_zero_pad_and_reject_silent_truncation():
    tokenizer = SimpleNamespace(encode=lambda prompt: SimpleNamespace(ids=[49406, 100, 49407]))
    tokens = sam3_onnx.text_tokens(tokenizer, "part")
    assert tokens.shape == (1, 32) and tokens.dtype == np.int64
    assert tokens[0, :4].tolist() == [49406, 100, 49407, 0]
    tokenizer.encode = lambda prompt: SimpleNamespace(ids=list(range(33)))
    with pytest.raises(ValueError, match="过长"):
        sam3_onnx.text_tokens(tokenizer, "long prompt")


def test_onnx_cuda_refuses_whole_model_cpu_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(sam3_onnx, "validate_bundle", lambda root: tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        preload_dlls=lambda: None, get_available_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=object))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    with pytest.raises(RuntimeError, match="未回退到 CPU"):
        sam3_onnx.Sam3OnnxRuntime(tmp_path, "cuda")


def test_sam3_candidates_filter_scores_and_reject_shifted_dimensions():
    masks = np.zeros((2, 1, 40, 50), dtype=bool)
    masks[:, :, 8:30, 12:39] = True
    result = mask_candidates(masks, [0.95, 0.1], 50, 40, 0.35)
    assert len(result["candidates"]) == 1
    assert result["width"] == 50 and result["height"] == 40
    with pytest.raises(RuntimeError, match="坐标偏移"):
        mask_candidates(masks, [0.95, 0.1], 100, 80, 0.35)
    with pytest.raises(RuntimeError, match="数量"):
        mask_candidates(masks, [0.95], 50, 40, 0.35)


def test_onnx_configuration_persists_backend_after_validation(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.write_bytes(b"test interpreter, never executed")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in (*sam3_onnx.MODEL_FILES, "tokenizer.json"):
        (bundle / name).write_bytes(b"test")
    assistant = Sam3Assistant(Store(tmp_path / "workspace"), object(), object())
    calls = []
    monkeypatch.setattr(assistant, "_validate_onnx", lambda *args: calls.append(args))
    monkeypatch.setattr(assistant, "_installed", lambda python, backend: backend == "onnx")
    result = assistant.configure({"backend": "onnx", "python_executable": str(python),
                                  "checkpoint_path": str(bundle), "device": "cuda"})
    assert calls == [(str(python), str(bundle))]
    assert result["backend"] == "onnx" and result["configured"] and result["installed"]
    assert assistant.config()["backend"] == "onnx"
    assert not result["running"]


def test_onnx_native_folder_picker_returns_path(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(native_picker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps([str(tmp_path)])))
    assert native_picker.select_images("sam3_onnx") == {"cancelled": False, "path": str(tmp_path)}


@pytest.mark.parametrize("result_size", [(80, 60), (160, 120)])
def test_sam3_api_checks_geometry_without_saving_annotations(tmp_path, monkeypatch, result_size):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (80, 60)).save(images / "frame.png")
    app = create_app(tmp_path / "workspace")
    summary = app.state.store.register({"name": "segment", "task_type": "segment",
                                        "image_root": str(images), "classes": ["part"]})
    project = app.state.store.project(summary["id"])
    item = project.items[0]["id"]
    before = app.state.store.detail(project.id, item)
    monkeypatch.setattr(app.state.sam3, "predict", lambda *args:
                        {"width": result_size[0], "height": result_size[1], "candidates": []})
    with TestClient(app) as client:
        response = client.post(f"/api/projects/{project.id}/items/{item}/sam3",
                               headers={"x-labelpilot-token": app.state.token}, json={"label": "part", "prompt": "part"})
    assert response.status_code == (200 if result_size == (80, 60) else 422)
    assert app.state.store.detail(project.id, item) == before
