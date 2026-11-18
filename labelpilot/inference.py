import copy
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from .store import Problem, atomic_json, digest, now, read_json


DISABLED_INTEGRATIONS = ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "dvc", "neptune", "raytune", "tensorboard")


def model_class_names(names, project_names):
    ordered = [names[index] for index in range(len(names))]
    if not ordered or ordered != project_names[:len(ordered)]:
        raise Problem("模型类别 ID / 名称与项目既有标签不一致；只允许在末尾追加新标签", 422)
    return ordered


class Inference:
    def __init__(self, store):
        self.store = store
        runtime = store.data / "ultralytics"
        runtime.mkdir(parents=True, exist_ok=True)
        matplotlib = store.data / "matplotlib"
        matplotlib.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(runtime)
        os.environ["MPLCONFIGDIR"] = str(matplotlib)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        self.lock = threading.Lock()
        self.exclusive = threading.Event()
        self.model = None
        self.signature = None
        self.model_digest = None
        self.device = None

    @contextmanager
    def prediction_slot(self):
        while not self.lock.acquire(timeout=0.1):
            if self.exclusive.is_set():
                raise Problem("本地训练或模型对比正在占用运行环境", 409)
        try:
            yield
        finally:
            self.lock.release()

    def release_cached_model(self):
        if self.model is not None:
            import torch
            self.model = self.signature = self.model_digest = self.device = None
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def predict(self, project_id, item_id, options):
        project = self.store.project(project_id)
        path = Path(project.config.get("model_path", ""))
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise Problem("未配置可信的本地 .pt 模型；不会联网下载权重", 422)
        confidence = options.get("confidence", 0.25)
        iou = options.get("iou", 0.7)
        image_size = options.get("image_size", 960)
        if not isinstance(confidence, (int, float)) or not 0.01 <= confidence <= 1:
            raise Problem("置信度必须在 0.01 到 1 之间", 422)
        if not isinstance(iou, (int, float)) or not 0.01 <= iou <= 1:
            raise Problem("IoU 必须在 0.01 到 1 之间", 422)
        if not isinstance(image_size, int) or image_size not in (640, 960, 1280):
            raise Problem("输入尺寸必须为 640、960 或 1280", 422)
        with self.prediction_slot():
            import torch
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS

            SETTINGS.update({name: False for name in DISABLED_INTEGRATIONS if name in SETTINGS})

            signature = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            if self.signature != signature:
                candidate = YOLO(str(path), task="detect")
                if candidate.task != "detect":
                    raise Problem("当前仅支持目标检测模型；不将分割或分类输出当成检测框", 422)
                model_class_names(candidate.names, project.names)
                self.model = candidate
                self.model_digest = digest(path)
                self.signature = signature
                self.device = 0 if torch.cuda.is_available() else "cpu"
            names = model_class_names(self.model.names, project.names)
            current = self.store.detail(project_id, item_id)
            item = project.item(item_id)
            with Image.open(item["path"]) as image:
                result = self.model.predict(source=image.convert("RGB"), conf=confidence, iou=iou,
                                            imgsz=image_size, device=self.device, max_det=300,
                                            verbose=False, save=False, half=False)[0]
            if tuple(result.orig_shape) != (current["height"], current["width"]):
                raise Problem("模型返回尺寸与原图不一致", 422)
            if not torch.isfinite(result.boxes.data).all().item():
                raise Problem("模型返回非有限坐标", 422)
            shapes = []
            for index, values in enumerate(result.boxes.data.detach().cpu().tolist()):
                left, top, right, bottom, score, class_id = values
                left, right = max(0, left), min(current["width"], right)
                top, bottom = max(0, top), min(current["height"], bottom)
                if right <= left or bottom <= top:
                    continue
                shapes.append({"label": names[int(class_id)], "score": score,
                               "points": [[left, top], [right, top], [right, bottom], [left, bottom]],
                               "shape_type": "rectangle", "group_id": None, "description": "",
                               "flags": {}, "attributes": {}, "difficult": False,
                               "prediction_origin": "model_prediction",
                               "prediction_id": f"{item_id}:{index}:{self.model_digest[:12]}"})
            provenance = {"model_path": str(path), "model_sha256": self.model_digest,
                          "class_names": names, "project_class_names": list(project.names),
                          "confidence": confidence, "iou": iou,
                          "image_size": image_size, "device": str(result.boxes.data.device),
                          "image_sha256": current["image_sha256"], "created_at": now(),
                          "human_reviewed": False, "speed_ms": result.speed}
            return {"shapes": shapes, "provenance": provenance, "base_version": current["version"],
                    "image_sha256": current["image_sha256"]}


class Jobs:
    def __init__(self, store, inference):
        self.store = store
        self.inference = inference
        self.lock = threading.Lock()
        self.records = {}
        self.cancel_events = {}
        self.directory = store.data / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*.json"):
            try:
                record = read_json(path)
                if record["status"] in ("queued", "running"):
                    record.update(status="interrupted", finished_at=now())
                    atomic_json(path, record)
                self.records[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue

    def update(self, job_id, **values):
        with self.lock:
            self.records[job_id].update(values)
            atomic_json(self.directory / f"{job_id}.json", self.records[job_id])

    def list(self, project_id):
        with self.lock:
            return sorted([copy.deepcopy(record) for record in self.records.values() if record["project"] == project_id],
                          key=lambda record: record["created_at"], reverse=True)[:20]

    def reserve_model_slot(self):
        with self.lock:
            if any(job["status"] in ("queued", "running") for job in self.records.values()):
                raise Problem("请等待批量预标注完成或先取消任务", 409)
            if not self.inference.lock.acquire(blocking=False):
                raise Problem("模型运行环境正在使用，请等待当前任务结束", 409)
            self.inference.exclusive.set()

    def release_model_slot(self):
        self.inference.exclusive.clear()
        self.inference.lock.release()

    def start(self, project_id, options):
        limit = options.get("limit", 100)
        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise Problem("每批数量必须为 1 到 10000", 422)
        listing = self.store.listing(project_id, query=options.get("query", ""), batch=options.get("batch", ""),
                                     status="unstarted", limit=10000)
        ids = [item["id"] for item in listing["items"][:limit]]
        if not ids:
            raise Problem("当前范围没有未开始图片；已有标注一律不会覆盖", 422)
        job_id = uuid.uuid4().hex
        with self.lock:
            if self.inference.exclusive.is_set():
                raise Problem("本地训练或模型对比正在占用运行环境", 409)
            if any(record["status"] in ("running", "queued") for record in self.records.values()):
                raise Problem("已有预标注任务运行中，请先完成或取消", 409)
            record = {"id": job_id, "project": project_id, "status": "queued", "total": len(ids),
                      "done": 0, "skipped": 0, "failed": 0, "errors": [], "created_at": now(), "options": options}
            self.records[job_id] = record
            self.cancel_events[job_id] = threading.Event()
            atomic_json(self.directory / f"{job_id}.json", record)
        threading.Thread(target=self.run, args=(job_id, ids), daemon=True).start()
        return copy.deepcopy(record)

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise Problem("任务不存在", 404)
            event = self.cancel_events.get(job_id)
            if event:
                event.set()
            return copy.deepcopy(self.records[job_id])

    def run(self, job_id, ids):
        record = copy.deepcopy(self.records[job_id])
        project_id = record["project"]
        self.update(job_id, status="running")
        done = skipped = failed = 0
        errors = []
        for item_id in ids:
            if self.cancel_events[job_id].is_set():
                self.update(job_id, status="cancelled", finished_at=now())
                return
            try:
                current = self.store.detail(project_id, item_id)
                if current["status"] != "unstarted":
                    skipped += 1
                else:
                    prediction = self.inference.predict(project_id, item_id, record["options"])
                    if self.cancel_events[job_id].is_set():
                        self.update(job_id, status="cancelled", finished_at=now())
                        return
                    document = copy.deepcopy(current["document"])
                    document["shapes"] = prediction["shapes"]
                    document["labelpilot"] = {"prediction": prediction["provenance"]}
                    self.store.save(project_id, item_id, {**current, "document": document,
                                                          "expected_version": current["version"], "status": "predicted"}, predicted=True)
                    done += 1
            except Problem as exc:
                if exc.status == 409:
                    skipped += 1
                else:
                    failed += 1
                    errors.append({"item": item_id, "error": exc.message})
            except Exception as exc:
                failed += 1
                errors.append({"item": item_id, "error": str(exc)[:300]})
            self.update(job_id, done=done, skipped=skipped, failed=failed, errors=errors[-50:])
            if failed >= 3 and done == 0:
                self.update(job_id, status="failed", finished_at=now())
                return
        self.update(job_id, status="completed_with_errors" if failed else "completed", finished_at=now())
