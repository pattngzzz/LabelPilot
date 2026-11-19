import copy
import math
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from .curation import save_findings
from .intake import rare_labels, route_review
from .store import AUDITABLE, PENDING_REVIEW, Problem, atomic_json, digest, now, read_json
from .tasks import HAND_COUNT, MAX_MASK_PIXELS, MODEL_TASK, mask_shape, model_labels, polygon_area


DISABLED_INTEGRATIONS = ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "dvc", "neptune", "raytune", "tensorboard")


def model_class_names(names, project_names, label_aliases=None):
    aliases = label_aliases or {}
    ordered = [aliases.get(names[index], names[index]) for index in range(len(names))]
    if not ordered or ordered != project_names[:len(ordered)]:
        raise Problem("模型类别 ID / 名称与项目既有标签不一致；仅允许末尾追加标签或显式配置的同 ID 更名", 422)
    return ordered


def validate_model_task(model, task):
    if model.task != MODEL_TASK[task]:
        raise Problem(f"项目需要 {MODEL_TASK[task]} 权重，当前模型为 {model.task}", 422)
    if task == "hand_pose" and tuple(getattr(model.model, "kpt_shape", ())) not in ((HAND_COUNT, 2), (HAND_COUNT, 3)):
        raise Problem("手部姿态需要 21 个手部关键点的权重", 422)


def prediction_shapes(result, task, names, width, height):
    if task == "semantic":
        import numpy as np

        mask = result.semantic_mask
        if mask is None or tuple(mask.data.shape) != (height, width):
            raise Problem("模型未返回原图大小的语义掩膜", 422)
        pixels = mask.data.detach().cpu().numpy()
        for index in np.unique(pixels):
            if index in (0, 255):
                continue
            if not 0 < index < len(names):
                raise Problem("掩膜中存在未知的类别 ID", 422)
            shape = mask_shape(Image.fromarray((pixels == index).astype("uint8") * 255))
            yield {**shape, "label": names[int(index)]}
        return

    if result.boxes is None:
        raise Problem("模型未返回有效的目标结果", 422)
    boxes = result.boxes.data.detach().cpu().tolist()
    polygons = None
    keypoints = None
    if task == "segment" and boxes:
        if result.masks is None or len(result.masks) != len(boxes):
            raise Problem("模型返回的实例掩膜与目标不匹配", 422)
        polygons = result.masks.xy
    if task == "hand_pose" and boxes:
        if result.keypoints is None or tuple(result.keypoints.data.shape[:2]) != (len(boxes), HAND_COUNT):
            raise Problem("模型未为每只手返回 21 个关键点", 422)
        keypoints = result.keypoints.data.detach().cpu().tolist()
    for index, values in enumerate(boxes):
        left, top, right, bottom, score, class_id = values
        if (not all(math.isfinite(value) for value in values)
                or not 0 <= class_id < len(names) or int(class_id) != class_id):
            raise Problem("模型返回无效的检测框或类别", 422)
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if right <= left or bottom <= top:
            continue
        shape = {"label": names[int(class_id)], "score": score, "shape_type": "rectangle",
                 "points": [[left, top], [right, top], [right, bottom], [left, bottom]]}
        if polygons is not None:
            if not all(math.isfinite(coord) for point in polygons[index] for coord in point):
                raise Problem("实例掩膜包含无效坐标", 422)
            points = [[min(width, max(0, float(x))), min(height, max(0, float(y)))] for x, y in polygons[index]]
            if len(points) < 3 or polygon_area(points) < 5e-9:
                continue
            shape.update(shape_type="polygon", points=points)
        if keypoints is not None:
            hand = []
            for x, y, *confidence in keypoints[index]:
                if not all(math.isfinite(value) for value in (x, y, *confidence)):
                    raise Problem("模型返回无效的手部关键点", 422)
                if confidence and confidence[0] < .25:
                    hand.append([0, 0, 0])
                    continue
                x, y = min(width, max(0, x)), min(height, max(0, y))
                hand.append([x, y, 1 if confidence and confidence[0] < .50 else 2])
                left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
            if not any(point[2] for point in hand):
                continue
            shape.update(shape_type="hand", points=[[left, top], [right, bottom]], keypoints=hand)
        yield shape


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

    def predict(self, project_id, item_id, options, *, model_path=None, model_sha256=None):
        project = self.store.project(project_id)
        task = project.task_type
        project_names = model_labels(task, project.names)
        path = Path(model_path or project.config.get("model_path", ""))
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise Problem("未配置可信的本地 .pt 模型；不会联网下载权重", 422)
        confidence = options.get("confidence", 0.25)
        iou = options.get("iou", 0.7)
        image_size = options.get("image_size", 960)
        if not isinstance(confidence, (int, float)) or not 0.01 <= confidence <= 1:
            raise Problem("置信度必须在 0.01 到 1 之间", 422)
        if not isinstance(iou, (int, float)) or not 0.01 <= iou <= 1:
            raise Problem("IoU 必须在 0.01 到 1 之间", 422)
        if not isinstance(image_size, int) or image_size not in (320, 640, 960, 1280):
            raise Problem("输入尺寸必须为 320、640、960 或 1280", 422)
        with self.prediction_slot():
            import torch
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS

            SETTINGS.update({name: False for name in DISABLED_INTEGRATIONS if name in SETTINGS})

            signature = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            if self.signature != signature:
                fingerprint = digest(path)
                if model_sha256 and model_sha256 != fingerprint:
                    raise Problem("工作流所选权重已变化，请恢复该版本或新建任务", 422)
                candidate = YOLO(str(path))
                validate_model_task(candidate, task)
                model_class_names(candidate.names, project_names, project.label_aliases)
                self.model = candidate
                self.model_digest = fingerprint
                self.signature = signature
                self.device = 0 if torch.cuda.is_available() else "cpu"
            if model_sha256 and self.model_digest != model_sha256:
                raise Problem("工作流所选权重已变化，请恢复该版本或新建任务", 422)
            validate_model_task(self.model, task)
            names = model_class_names(self.model.names, project_names, project.label_aliases)
            current = self.store.detail(project_id, item_id)
            width, height = current["width"], current["height"]
            if task == "semantic" and width * height > MAX_MASK_PIXELS:
                raise Problem("语义掩膜最多支持 3200 万像素", 422)
            item = project.item(item_id)
            with Image.open(item["path"]) as image:
                result = self.model.predict(source=image.convert("RGB"), conf=confidence, iou=iou,
                                            imgsz=image_size, device=self.device, max_det=300,
                                            verbose=False, save=False, half=False)[0]
            if tuple(result.orig_shape) != (height, width):
                raise Problem("模型返回尺寸与原图不一致", 422)
            shapes = [{**shape, "group_id": None, "description": "", "flags": {}, "attributes": {},
                       "difficult": False, "prediction_origin": "model_prediction",
                       "prediction_id": f"{item_id}:{index}:{self.model_digest[:12]}"}
                      for index, shape in enumerate(prediction_shapes(result, task, names, width, height))]
            output = result.semantic_mask if task == "semantic" else result.boxes
            provenance = {"model_path": str(path), "model_sha256": self.model_digest,
                          "class_names": names, "project_class_names": list(project.names),
                          "task_type": task, "confidence": confidence, "iou": iou,
                          "image_size": image_size, "device": str(output.data.device),
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

    def lookup(self, project_id, job_id):
        with self.lock:
            record = self.records.get(job_id)
            return copy.deepcopy(record) if record and record["project"] == project_id else None

    def reserve_model_slot(self):
        with self.lock:
            if any(job["status"] in ("queued", "running") for job in self.records.values()):
                raise Problem("请等待批量预标注或疑点扫描完成，或先取消任务", 409)
            if not self.inference.lock.acquire(blocking=False):
                raise Problem("模型运行环境正在使用，请等待当前任务结束", 409)
            self.inference.exclusive.set()

    def release_model_slot(self):
        self.inference.exclusive.clear()
        self.inference.lock.release()

    def start(self, project_id, options, *, audit=False, routing=False, task_id=None, pinned_model=None):
        project = self.store.project(project_id)
        limit = options.get("limit", 100)
        query, batch = options.get("query", ""), options.get("batch", "")
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise Problem("每批数量必须为 1 到 10000", 422)
        if not isinstance(query, str) or not isinstance(batch, str):
            raise Problem("批次和关键词必须为文本", 422)
        if audit:
            if project.task_type != "detect":
                raise Problem("按框查找标注疑点适用于目标检测项目", 422)
            model = Path(project.config.get("model_path") or "")
            if model.suffix.lower() != ".pt" or not model.is_file():
                raise Problem("请在模型设置中保存一个本地检测模型，再扫描标注疑点", 422)
            options = {**options, "confidence": options.get("confidence", .60)}
            if type(options["confidence"]) not in (int, float) or not .05 <= options["confidence"] <= 1:
                raise Problem("疑点置信度需在 0.05 到 1 之间", 422)
        status = "review" if routing else "auditable" if audit else "curated" if options.get("curated") is True else "unstarted"
        listing = self.store.listing(project_id, query=query, batch=batch, status=status,
                                     limit=len(project.items) if routing else limit)
        routes = self.store.review_routes(project_id) if routing else {}
        ids = []
        for item in listing["items"]:
            route = routes.get(item["id"], {})
            if routing and route.get("version") == item.get("version", 0) and route.get("key") == options["route_key"]:
                continue
            ids.append(item["id"])
            if len(ids) == limit:
                break
        if not ids:
            if routing:
                raise Problem("当前没有需要重新分流的候选图", 422)
            raise Problem("当前范围没有可复核的人工标注" if audit else "当前范围没有未开始图片；已有标注一律不会覆盖", 422)
        job_id = task_id or uuid.uuid4().hex
        with self.lock:
            if self.inference.exclusive.is_set():
                raise Problem("本地训练或模型对比正在占用运行环境", 409)
            if any(record["status"] in ("running", "queued") for record in self.records.values()):
                raise Problem("已有批量模型任务运行中，请先完成或取消", 409)
            record = {"id": job_id, "project": project_id, "status": "queued", "total": len(ids),
                      "done": 0, "skipped": 0, "failed": 0, "errors": [], "created_at": now(), "options": options}
            if audit:
                record.update(kind="audit", flagged=0)
            if routing:
                record.update(kind="routing", flagged=0)
            if pinned_model:
                record["pinned_model"] = pinned_model
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
        audit = record.get("kind") == "audit"
        routing = record.get("kind") == "routing"
        available = PENDING_REVIEW if routing else AUDITABLE if audit else ("unstarted",)
        options = record["options"]
        model = record.get("pinned_model", {})
        rare = rare_labels(self.store, project_id, options["rare"]) if routing else set()
        done = skipped = failed = flagged = 0
        errors = []
        for item_id in ids:
            if self.cancel_events[job_id].is_set():
                self.update(job_id, status="cancelled", finished_at=now())
                return
            try:
                current = self.store.detail(project_id, item_id)
                if current["status"] not in available:
                    skipped += 1
                else:
                    if audit:
                        self.store.validate(self.store.project(project_id), current["document"], current["width"],
                                            current["height"], "draft", False)
                    prediction = None
                    if model:
                        prediction = self.inference.predict(project_id, item_id, options,
                                                            model_path=model["path"], model_sha256=model["sha256"])
                    elif not routing:
                        prediction = self.inference.predict(project_id, item_id, options)
                    if self.cancel_events[job_id].is_set():
                        self.update(job_id, status="cancelled", finished_at=now())
                        return
                    if routing:
                        flagged += route_review(self.store, project_id, item_id, current, prediction, rare,
                                                options["low"], options["route_key"])
                    elif audit:
                        flagged += save_findings(self.store, project_id, item_id, current, prediction)
                    else:
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
            progress = {"done": done, "skipped": skipped, "failed": failed, "errors": errors[-50:]}
            if audit or routing:
                progress["flagged"] = flagged
            self.update(job_id, **progress)
            if failed >= 3 and done == 0:
                self.update(job_id, status="failed", finished_at=now())
                return
        self.update(job_id, status="completed_with_errors" if failed else "completed", finished_at=now())
