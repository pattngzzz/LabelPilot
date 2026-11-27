import json
import hashlib
import math
import os
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from .ocr import validate_ocr_shape
from .paddleocr_models import DEFAULT_MODELS, MODEL_ROLES, model_files, model_signature
from .store import Problem, atomic_json, digest


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

    def _read_configuration(self):
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise Problem("尚未配置本地 PaddleOCR；请检查 data/ocr_runtime.json", 503) from exc
        except (OSError, ValueError) as exc:
            raise Problem("本地 PaddleOCR 配置无法读取", 503) from exc
        if not isinstance(value, dict):
            raise Problem("本地 PaddleOCR 配置格式无效", 503)
        return value

    def _configuration(self, value=None, *, validate_models=True):
        value = self._read_configuration() if value is None else value
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
        models = {}
        for role, key in MODEL_ROLES.items():
            selected = value.get(key)
            if validate_models and selected is not None and selected != "":
                try:
                    models[role] = str(model_files(selected)[0].parent)
                except (ValueError, OSError) as exc:
                    raise Problem(f"本地 OCR {role} 模型配置无效：{exc}", 503) from exc
        return python.resolve(), cache.resolve(), model_directory.resolve(), device, models

    def status(self):
        try:
            configuration = self._read_configuration()
            python, cache, model_directory, device, models = self._configuration(configuration)
        except Problem as exc:
            return {"configured": False, "runtime_available": False, "models_downloaded": False,
                    "device": None, "model_cache": None, "message": exc.message}
        available = python.is_file() and self.worker_path.is_file()
        downloaded = all(role in models or (cache / "official_models" / name / "inference.pdiparams").is_file()
                         or (model_directory / "official_models" / name / "inference.pdiparams").is_file()
                         for role, name in DEFAULT_MODELS.items())
        message = "本地OCR运行环境可用" if available else "未找到独立OCR Python环境"
        if available and not downloaded:
            message = "已配置；首次预标注将下载模型到本机目录"
        if models:
            message = "已配置自选 OCR 模型；首次预标注将检查实际加载兼容性" if available else message
        return {"configured": True, "runtime_available": available, "models_downloaded": downloaded,
                "worker_running": bool(self.process and self.process.poll() is None), "device": device,
                "model_cache": str(cache), "model_directory": str(model_directory),
                "detection_model": configuration.get("text_detection_model_name", DETECTION_MODEL) if "detection" in models else DETECTION_MODEL,
                "recognition_model": configuration.get("text_recognition_model_name", RECOGNITION_MODEL) if "recognition" in models else RECOGNITION_MODEL,
                **{key: models.get(role) for role, key in MODEL_ROLES.items()}, "message": message}

    def set_device(self, device):
        return self.configure({"device": device})

    def _inspect_models(self, python, models):
        try:
            result = subprocess.run([str(python), "-X", "utf8", str(self.worker_path), "--inspect-models"],
                                    input=json.dumps(models, ensure_ascii=True), capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=60,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            response = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise Problem("无法校验 OCR 模型，请检查独立 OCR 环境与推理目录", 422) from exc
        if not isinstance(response, dict):
            raise Problem("OCR 模型校验进程返回了无效响应", 422)
        if result.returncode or not response.get("ok"):
            raise Problem(f"OCR 模型校验失败：{response.get('error', '独立环境不可用')}", 422)
        return response["models"]

    def configure(self, payload):
        if not isinstance(payload, dict) or not payload or set(payload) - {"device", *MODEL_ROLES.values()}:
            raise Problem("请选择 OCR 设备或检测 / 识别模型目录", 422)
        if "device" in payload and payload["device"] not in ("gpu:0", "cpu"):
            raise Problem("OCR 运行设备仅支持 GPU 或 CPU", 422)
        with self.lock:
            value = self._read_configuration()
            python, _, _, _, _ = self._configuration(value, validate_models=False)
            updated = dict(value)
            for key, selected in payload.items():
                if key == "device":
                    updated[key] = selected
                elif selected is None or selected == "":
                    updated.pop(key, None)
                    updated.pop(key.replace("_dir", "_name"), None)
                else:
                    try:
                        updated[key] = str(model_files(selected)[0].parent)
                    except (ValueError, OSError) as exc:
                        raise Problem(str(exc), 422) from exc
            models = {role: updated[key] for role, key in MODEL_ROLES.items() if updated.get(key)}
            if set(payload) & set(MODEL_ROLES.values()) and models:
                inspected = self._inspect_models(python, models)
                for role, spec in inspected.items():
                    updated[MODEL_ROLES[role].replace("_dir", "_name")] = spec["name"]
            if updated != value:
                atomic_json(self.config_path, updated)
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

    def _start_worker(self, python, cache, model_directory, device, models):
        if not python.is_file():
            raise Problem("找不到独立 PaddleOCR Python 环境", 503)
        if not self.worker_path.is_file():
            raise Problem("LabelPilot 的 PaddleOCR worker 文件缺失", 503)
        signature = self._signature(python, cache, model_directory, device, models)
        cache.mkdir(parents=True, exist_ok=True)
        for role, model_name in DEFAULT_MODELS.items():
            if role in models:
                continue
            self._link_model_files(model_directory / "official_models" / model_name,
                                   cache / "official_models" / model_name)
        inspected = self._inspect_models(python, models) if models else {}
        staged = {}
        for role, spec in inspected.items():
            version = hashlib.sha256(json.dumps(spec["sha256"], sort_keys=True).encode()).hexdigest()
            destination = cache / "custom_models" / f"{role}_{version}"
            destination.mkdir(parents=True, exist_ok=True)
            for source in model_files(spec["directory"]):
                target = destination / source.name
                if not target.is_file() or digest(target) != spec["sha256"][source.name]:
                    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
                    try:
                        shutil.copy2(source, temporary)
                        if digest(temporary) != spec["sha256"][source.name]:
                            raise Problem("复制期间 OCR 权重发生变化，请重试", 409)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
            staged[role] = {**spec, "directory": str(destination), "source_directory": spec["directory"]}
        if self._signature(python, cache, model_directory, device, models) != signature:
            raise Problem("准备期间 OCR 权重发生变化，请重试", 409)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab")
        environment = os.environ.copy()
        environment["PADDLE_PDX_CACHE_HOME"] = str(cache)
        environment["PYTHONUTF8"] = "1"
        environment["LABELPILOT_OCR_DEVICE"] = device
        environment["LABELPILOT_OCR_MODELS"] = json.dumps(staged, ensure_ascii=True)
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
        self.signature = signature
        threading.Thread(target=self._read_responses, args=(process, self.responses), daemon=True).start()

    @staticmethod
    def _signature(python, cache, model_directory, device, models):
        return (str(python), str(cache), str(model_directory), device,
                tuple((role, model_signature(path)) for role, path in sorted(models.items())))

    def predict(self, image_path, *, threshold=0.3):
        with self.lock:
            python, cache, model_directory, device, models = self._configuration()
            signature = self._signature(python, cache, model_directory, device, models)
            if self.process is None or self.process.poll() is not None or self.signature != signature:
                self._stop_worker()
                self._start_worker(python, cache, model_directory, device, models)
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
                    for role, model_name in DEFAULT_MODELS.items():
                        if role in models:
                            continue
                        self._link_model_files(cache / "official_models" / model_name,
                                               model_directory / "official_models" / model_name)
                return response.get("detections", []), response.get("runtime", {})

    def close(self):
        with self.lock:
            self._stop_worker()
