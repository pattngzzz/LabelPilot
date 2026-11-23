import json
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .store import Problem, atomic_json, read_json


class Sam3Assistant:
    def __init__(self, store, inference, jobs):
        self.inference = inference
        self.jobs = jobs
        self.config_path = store.data / "sam3" / "config.json"
        self.log_path = store.data / "sam3" / "worker.log"
        self.lock = threading.RLock()
        self.process = self.port = self.token = None
        self.slot_reserved = False

    def config(self):
        value = read_json(self.config_path) if self.config_path.is_file() else {}
        return value if isinstance(value, dict) else {}

    def _paths(self, value):
        python, checkpoint = value.get("python_executable", ""), value.get("checkpoint_path", "")
        device = value.get("device", "cuda")
        backend = value.get("backend", "pytorch")
        if not isinstance(python, str) or not isinstance(checkpoint, str) or device not in ("cuda", "cpu") or backend not in ("pytorch", "onnx"):
            raise Problem("SAM3 Python、权重路径或设备设置无效", 422)
        if not python.strip() or not checkpoint.strip():
            raise Problem("请先选择有效的 SAM3 Python 环境和本地权重文件", 422)
        try:
            python, checkpoint = Path(python).expanduser().resolve(strict=True), Path(checkpoint).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise Problem("请先选择有效的 SAM3 Python 环境和本地权重文件", 422)
        if not python.is_file() or python.name.lower() not in ("python.exe", "python"):
            raise Problem("所选 SAM3 Python 必须是独立环境中的 python.exe", 422)
        if backend == "onnx":
            if not checkpoint.is_dir() or any(not (checkpoint / name).is_file() for name in
                                              ("sam3_image_encoder.onnx", "sam3_language_encoder.onnx", "sam3_decoder.onnx", "tokenizer.json")):
                raise Problem("请选择包含三个 SAM3 ONNX 模型与 tokenizer.json 的文件夹", 422)
        elif not checkpoint.is_file() or checkpoint.suffix.lower() not in (".pt", ".pth"):
            raise Problem("请选择本地 SAM3 .pt 或 .pth 权重文件", 422)
        return str(python), str(checkpoint), device

    def configure(self, payload):
        with self.lock:
            if any(record.get("kind") == "sam3_contours" and record["status"] in ("queued", "running")
                   for record in getattr(self.jobs, "records", {}).values()):
                raise Problem("请先完成或取消批量轮廓任务，再修改 SAM3 配置", 409)
            self._reap_worker()
            if self.process is not None:
                raise Problem("请先释放 SAM3 显存，再修改配置", 409)
            python, checkpoint, device = self._paths(payload)
            backend = payload.get("backend", "pytorch")
            if backend == "onnx":
                self._validate_onnx(python, checkpoint)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.config_path, {"python_executable": python, "checkpoint_path": checkpoint, "device": device,
                                           "backend": backend})
            return self.status()

    def _validate_onnx(self, python, checkpoint):
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        try:
            result = subprocess.run([python, "-X", "utf8", "-m", "labelpilot.sam3_worker", "--check",
                                     "--backend", "onnx", "--checkpoint", checkpoint],
                                    cwd=Path(__file__).parent.parent, capture_output=True, timeout=45,
                                    encoding="utf-8", errors="replace", env=environment,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (OSError, subprocess.SubprocessError):
            raise Problem("SAM3 ONNX 完整性检查失败，请检查独立环境", 503)
        if result.returncode:
            detail = result.stderr.strip().splitlines()
            raise Problem("SAM3 ONNX 校验失败：" + (detail[-1][-400:] if detail else "请检查文件与依赖"), 422)

    def _installed(self, python, backend="pytorch"):
        if not python or not Path(python).is_file():
            return False
        try:
            environment = os.environ.copy()
            environment["PYTHONNOUSERSITE"] = "1"
            modules = ("onnxruntime", "onnx", "tokenizers", "cv2", "numpy", "PIL") if backend == "onnx" else ("sam3",)
            code = f"import importlib.util,sys; sys.exit(not all(importlib.util.find_spec(name) for name in {modules!r}))"
            result = subprocess.run([python, "-c", code],
                                    capture_output=True, timeout=15,
                                    env=environment,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _health(self, timeout=1):
        if not self.port or not self.token:
            return None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/health", headers={"X-LabelPilot-SAM3": self.token})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def _reap_worker(self):
        if self.process is not None and self.process.poll() is not None:
            self.process = self.port = self.token = None
            if self.slot_reserved:
                self.jobs.release_model_slot()
                self.slot_reserved = False

    def status(self):
        with self.lock:
            self._reap_worker()
            config = self.config()
            backend = config.get("backend", "pytorch")
            python, checkpoint = config.get("python_executable", ""), config.get("checkpoint_path", "")
            python_exists = isinstance(python, str) and Path(python).is_file()
            checkpoint_exists = bool(isinstance(checkpoint, str) and checkpoint and
                                     (Path(checkpoint).is_dir() if backend == "onnx" else Path(checkpoint).is_file()))
            installed = self._installed(python, backend) if python_exists else False
            health = self._health() if self.process is not None else None
            configured = bool(python_exists and checkpoint_exists)
            if not python_exists:
                message = "选择 SAM3 独立 Python 环境和本地权重文件后即可使用。"
            elif not installed:
                message = "所选 Python 环境未安装 SAM3 所需依赖；请按 README 安装隔离环境。"
            elif not checkpoint_exists:
                message = "已连接隔离 SAM3 环境；请选择本机获授权的 checkpoint。"
            elif health and health.get("ready"):
                message = "SAM3 已加载；正在独占模型运行环境，YOLO 推理和训练暂不可用。"
            elif self.process is not None:
                message = "SAM3 正在加载；首次启动可能需要几分钟。"
            else:
                message = "SAM3 已配置，尚未加载模型。"
            return {"configured": configured, "installed": installed, "running": self.process is not None,
                    "ready": bool(health and health.get("ready")), "device": config.get("device", "cuda"), "backend": backend,
                    "python_executable": python if isinstance(python, str) else "",
                    "checkpoint_path": checkpoint if isinstance(checkpoint, str) else "",
                    "message": message}

    def _reserve_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _start_worker(self, owner_job_id=None):
        self._reap_worker()
        if self.process is not None and (self._health() or {}).get("ready"):
            return
        python, checkpoint, device = self._paths(self.config())
        backend = self.config().get("backend", "pytorch")
        if not self._installed(python, backend):
            raise Problem("SAM3 尚未安装在所选 Python 环境；请按 README 安装隔离环境", 503)
        self.jobs.reserve_model_slot(owner_job_id=owner_job_id)
        self.slot_reserved = True
        try:
            self.inference.release_cached_model()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.port, self.token = self._reserve_port(), secrets.token_urlsafe(32)
            environment = os.environ.copy()
            environment["LABELPILOT_SAM3_TOKEN"] = self.token
            environment["LABELPILOT_SAM3_DEVICE"] = device
            environment["PYTHONNOUSERSITE"] = "1"
            with self.log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    [python, "-X", "utf8", "-m", "labelpilot.sam3_worker", "--backend", backend,
                     "--checkpoint", checkpoint, "--port", str(self.port)],
                    cwd=Path(__file__).parent.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise Problem("SAM3 启动失败；请检查 data/sam3/worker.log 和 CUDA / 权重配置", 503)
                health = self._health(timeout=2)
                if health and health.get("ready"):
                    return
                time.sleep(0.5)
            raise Problem("SAM3 加载超时；请检查 data/sam3/worker.log", 504)
        except Exception:
            self._stop_worker()
            raise

    def predict(self, image_path, payload, *, owner_job_id=None):
        prompt = payload.get("prompt")
        targets = payload.get("targets")
        threshold = payload.get("threshold", 0.35)
        if targets is None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 160):
            raise Problem("请输入 1 到 160 个字符的对象描述", 422)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.05 <= threshold <= 0.95:
            raise Problem("置信度阈值必须在 0.05 到 0.95 之间", 422)
        with self.lock:
            active = list(getattr(self.jobs, "records", {}).values())
            if any(job["status"] in ("queued", "running") and job["id"] != owner_job_id for job in active):
                raise Problem("请等待当前批量任务完成或取消后再推理", 409)
            self._start_worker(owner_job_id)
            body = json.dumps({"image_path": str(Path(image_path).resolve()), "prompt": prompt.strip() if isinstance(prompt, str) else "",
                               **({"targets": targets} if targets is not None else {}),
                               "threshold": float(threshold)}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(f"http://127.0.0.1:{self.port}/predict", data=body,
                                             headers={"Content-Type": "application/json",
                                                      "X-LabelPilot-SAM3": self.token}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                try:
                    detail = json.loads(exc.read().decode("utf-8")).get("detail", "SAM3 推理失败")
                except (ValueError, AttributeError):
                    detail = "SAM3 推理失败"
                raise Problem(detail, exc.code if 400 <= exc.code < 500 else 503)
            except (OSError, TimeoutError, ValueError):
                self._stop_worker()
                raise Problem("SAM3 服务连接失败；请检查 data/sam3/worker.log", 503)

    def _stop_worker(self):
        process = self.process
        if process is not None and process.poll() is None:
            request = urllib.request.Request(f"http://127.0.0.1:{self.port}/shutdown",
                                             headers={"X-LabelPilot-SAM3": self.token}, method="POST")
            try:
                urllib.request.urlopen(request, timeout=1).close()
            except OSError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.process = self.port = self.token = None
        if self.slot_reserved:
            self.jobs.release_model_slot()
            self.slot_reserved = False

    def release(self):
        with self.lock:
            if any(record.get("kind") == "sam3_contours" and record["status"] in ("queued", "running")
                   for record in getattr(self.jobs, "records", {}).values()):
                raise Problem("请先取消批量轮廓任务，等待当前图片完成后再释放显存", 409)
            self._stop_worker()
            return self.status()

    def shutdown(self):
        with self.lock:
            self._stop_worker()
