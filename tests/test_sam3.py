import numpy as np
import pytest
from fastapi.testclient import TestClient

from labelpilot.sam3 import Sam3Assistant
from labelpilot.sam3_worker import mask_polygon
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
