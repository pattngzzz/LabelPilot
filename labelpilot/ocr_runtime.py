import json
import math
import os
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from .ocr import validate_ocr_shape
from .store import Problem, atomic_json


DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "PP-OCRv5_server_rec"
WORKER_TIMEOUT_SECONDS = 300


def build_ocr_shapes(detections, label, width, height, item_id):
    if not isinstance(detections, list):
        raise Problem("PaddleOCR 未返回有效文字区域", 422)
    shapes = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        raw_points = detection.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            continue
        try:
            points = []
            for point in raw_points:
                x, y = float(point[0]), float(point[1])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("non-finite point")
                points.append([min(width, max(0.0, x)), min(height, max(0.0, y))])
        except (IndexError, TypeError, ValueError):
            continue
        center_x = sum(point[0] for point in points) / 4
        center_y = sum(point[1] for point in points) / 4
        points.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        text = detection.get("text", "")
        if not isinstance(text, str):
            text = ""
        text = " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())
        if len(text) > 4096:
            text = ""
        try:
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0
        shape = {
            "label": label, "shape_type": "polygon", "points": points, "text_raw": text,
            "readability": "readable" if text else "unreadable", "value_role": "unknown",
            "value_decimal": None, "unit": None, "field_name": None, "channel": None,
            "parent_id": None, "score": score, "text_recognition_confidence": score,
            "prediction_origin": "paddleocr_prediction",
            "prediction_id": f"{item_id}:paddleocr:{index}",
        }
        try:
            validate_ocr_shape(shape)
        except ValueError:
            continue
        shapes.append(shape)
    return shapes


class PaddleOCRRuntime:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.config_path = self.home / "data" / "ocr_runtime.json"
        self.log_path = self.home / "data" / "ocr_worker.log"
        self.worker_path = Path(__file__).with_name("paddleocr_worker.py")
        self.lock = threading.Lock()
        self.process = self.responses = self.signature = self.log_handle = None

    def _configuration(self):
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise Problem("尚未配置本地 PaddleOCR；请检查 data/ocr_runtime.json", 503) from exc
        except (OSError, ValueError) as exc:
            raise Problem("本地 PaddleOCR 配置无法读取", 503) from exc
        if not isinstance(value, dict):
            raise Problem("本地 PaddleOCR 配置格式无效", 503)
        path_values = (value.get("python_executable"), value.get("model_cache"), value.get("model_directory"))
        if any(not isinstance(path, str) or not path.strip() for path in path_values):
            raise Problem("本地 PaddleOCR 配置路径或设备无效", 503)
        try:
            python, cache, model_directory = (Path(path).expanduser() for path in path_values)
        except (TypeError, ValueError) as exc:
            raise Problem("本地 PaddleOCR 配置路径或设备无效", 503) from exc
        device = value.get("device", "gpu:0")
        if (not python.is_absolute() or not cache.is_absolute() or not model_directory.is_absolute()
                or device not in ("gpu:0", "cpu")):
            raise Problem("本地 PaddleOCR 配置路径或设备无效", 503)
        return python.resolve(), cache.resolve(), model_directory.resolve(), device

    def status(self):
        try:
            python, cache, model_directory, device = self._configuration()
        except Problem as exc:
            return {"configured": False, "runtime_available": False, "models_downloaded": False,
                    "device": None, "model_cache": None, "message": exc.message}
        available = python.is_file() and self.worker_path.is_file()
        downloaded = (cache / ".models_ready").is_file()
        message = "本地OCR运行环境可用" if available else "未找到独立OCR Python环境"
        if available and not downloaded:
            message = "已配置；首次预标注将下载模型到本机目录"
        return {"configured": True, "runtime_available": available, "models_downloaded": downloaded,
                "worker_running": bool(self.process and self.process.poll() is None), "device": device,
                "model_cache": str(cache), "model_directory": str(model_directory),
                "detection_model": DETECTION_MODEL,
                "recognition_model": RECOGNITION_MODEL, "message": message}

    def set_device(self, device):
        if device not in ("gpu:0", "cpu"):
            raise Problem("OCR 运行设备仅支持 GPU 或 CPU", 422)
        with self.lock:
            try:
                value = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise Problem("本地 OCR 配置不存在或无法读取", 503) from exc
            value["device"] = device
            atomic_json(self.config_path, value)
            self._stop_worker()
        return self.status()

    def _read_responses(self, process, responses):
        for line in process.stdout:
            try:
                responses.put(json.loads(line))
            except ValueError:
                responses.put({"protocol_error": "OCR worker 返回了无效响应"})
        responses.put(None)

    def _stop_worker(self):
        process, self.process = self.process, None
        self.signature = self.responses = None
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            for pipe in (process.stdin, process.stdout):
                try:
                    pipe.close()
                except (OSError, AttributeError):
                    pass
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    @staticmethod
    def _link_model_files(source, destination):
        if not source.is_dir():
            return
        for source_file in source.rglob("*"):
            if not source_file.is_file():
                continue
            target_file = destination / source_file.relative_to(source)
            if target_file.exists():
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_file, target_file)
            except OSError:
                shutil.copy2(source_file, target_file)

    def _start_worker(self, python, cache, model_directory, device):
        if not python.is_file():
            raise Problem("找不到独立 PaddleOCR Python 环境", 503)
        if not self.worker_path.is_file():
            raise Problem("LabelPilot 的 PaddleOCR worker 文件缺失", 503)
        cache.mkdir(parents=True, exist_ok=True)
        for model_name in (DETECTION_MODEL, RECOGNITION_MODEL):
            self._link_model_files(model_directory / "official_models" / model_name,
                                   cache / "official_models" / model_name)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab")
        environment = os.environ.copy()
        environment["PADDLE_PDX_CACHE_HOME"] = str(cache)
        environment["PYTHONUTF8"] = "1"
        environment["LABELPILOT_OCR_DEVICE"] = device
        try:
            process = subprocess.Popen(
                [str(python), "-X", "utf8", "-u", str(self.worker_path)], cwd=str(self.home),
                env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_handle,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.log_handle.close()
            self.log_handle = None
            raise Problem(f"无法启动 PaddleOCR 独立进程：{exc}", 503) from exc
        self.process = process
        self.responses = queue.Queue()
        self.signature = (str(python), str(cache), str(model_directory), device)
        threading.Thread(target=self._read_responses, args=(process, self.responses), daemon=True).start()

    def predict(self, image_path, *, threshold=0.3):
        with self.lock:
            python, cache, model_directory, device = self._configuration()
            signature = (str(python), str(cache), str(model_directory), device)
            if self.process is None or self.process.poll() is not None or self.signature != signature:
                self._stop_worker()
                self._start_worker(python, cache, model_directory, device)
            request_id = uuid.uuid4().hex
            request = {"id": request_id, "command": "predict", "image_path": str(Path(image_path).resolve()),
                       "text_det_thresh": threshold}
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
            except (OSError, BrokenPipeError) as exc:
                self._stop_worker()
                raise Problem("PaddleOCR worker 已退出，请查看 data/ocr_worker.log", 503) from exc
            while True:
                try:
                    response = self.responses.get(timeout=WORKER_TIMEOUT_SECONDS)
                except queue.Empty as exc:
                    self._stop_worker()
                    raise Problem("PaddleOCR 推理超时，请查看 data/ocr_worker.log", 504) from exc
                if response is None:
                    self._stop_worker()
                    raise Problem("PaddleOCR worker 提前退出，请查看 data/ocr_worker.log", 503)
                if response.get("protocol_error"):
                    self._stop_worker()
                    raise Problem(response["protocol_error"], 503)
                if response.get("id") != request_id:
                    continue
                if not response.get("ok"):
                    raise Problem(f"PaddleOCR 推理失败：{response.get('error', '未知错误')}", 503)
                if response.get("models_ready"):
                    (cache / ".models_ready").touch()
                    for model_name in (DETECTION_MODEL, RECOGNITION_MODEL):
                        self._link_model_files(cache / "official_models" / model_name,
                                               model_directory / "official_models" / model_name)
                return response.get("detections", []), response.get("runtime", {})

    def close(self):
        with self.lock:
            self._stop_worker()
