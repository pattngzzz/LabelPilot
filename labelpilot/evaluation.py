import copy
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from pathlib import Path

from .blindsets import ID, get_blindset, verified_manifest
from .comparison_details import DETAILS_FILE
from .store import Problem, atomic_json, digest, now
from .training import LocalJobCancelled, has_training_dependencies, read_state, run_local_worker


ACTIVE_COMPARISON = ("preparing", "running")


def iou(first, second):
    a, b, c, d = first
    x, y, z, w = second
    intersection = max(0, min(c, z) - max(a, x)) * max(0, min(d, w) - max(b, y))
    union = (c - a) * (d - b) + (z - x) * (w - y) - intersection
    return intersection / union if union > 0 else 0


def count_detections(counts, labels, detections, threshold):
    unmatched = defaultdict(list)
    for index, (name, _) in enumerate(labels):
        unmatched[name].append(index)
        counts[name]["ground_truth"] += 1
        counts[name]["missed"] += 1

    predictions = sorted((row for row in detections if row[0] in counts), key=lambda row: row[1], reverse=True)
    details = []
    for name, confidence, box in predictions:
        row = counts[name]
        available = unmatched[name]
        match, overlap = None, None
        if available:
            candidate, score = max(((index, iou(labels[index][1], box)) for index in available), key=lambda entry: entry[1])
            if score >= threshold:
                match, overlap = candidate, score
                available.remove(match)
        if match is None:
            row["false_positives"] += 1
        else:
            row["matched"] += 1
            row["missed"] -= 1
        details.append({"label": name, "confidence": confidence, "box": box, "truth_index": match, "iou": overlap})
    return details


def comparison_class_names(names, classes, label_aliases=None):
    aliases = label_aliases or {}
    ordered = [aliases.get(names[index], names[index]) for index in range(len(names))]
    shared = min(len(ordered), len(classes))
    if not shared or ordered[:shared] != classes[:shared]:
        raise Problem("模型类别 ID / 名称与盲测集不匹配", 422)
    return ordered


def comparison_options(payload):
    confidence, match_iou = payload.get("confidence", .25), payload.get("match_iou", .5)
    image_size = payload.get("image_size", 640)
    if any(type(value) not in (int, float) or not .01 <= value <= 1 for value in (confidence, match_iou)):
        raise Problem("置信度和匹配 IoU 必须在 0.01 到 1 之间", 422)
    if type(image_size) is not int or image_size not in (320, 640, 960, 1280):
        raise Problem("请选择 320、640、960 或 1280 的输入尺寸", 422)
    return {"confidence": confidence, "match_iou": match_iou, "image_size": image_size, "nms_iou": .7, "max_det": 300}


def weight_path(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Problem("请选择两个可信的本地 .pt 检测权重", 422)
    try:
        path = Path(value.strip()).expanduser()
        if path.is_absolute() and path.suffix.lower() == ".pt" and path.is_file():
            return path.resolve()
    except (OSError, ValueError, RuntimeError):
        pass
    raise Problem("请选择可读取的本地 .pt 文件", 422)


def recorded_overlap(store, project_id, path, blindset, manifest):
    training = (store.data / "training").resolve()
    if not path.is_relative_to(training):
        return None
    parts = path.relative_to(training).parts
    if len(parts) != 4 or parts[1:] != ("run", "weights", "best.pt"):
        return None
    directory = training / parts[0]
    job = read_state(directory / "job.json")
    dataset = read_state(directory / "dataset" / "export_manifest.json")
    if job.get("project") != project_id or job.get("status") != "completed" or not isinstance(dataset.get("items"), list):
        return None
    blind_ids = {row["id"] for row in manifest["items"]}
    blind_hashes = {row["sha256"] for row in manifest["items"]}
    groups = set(blindset["source_groups"])
    seen = dataset["items"]
    directories = set()
    if (dataset.get("split_config") or {}).get("unit") == "directory":
        directories = set(blindset["folders"]).intersection(row.get("split_group") for row in seen)
    return {"images": sum(row["id"] in blind_ids or row["sha256"] in blind_hashes for row in seen),
            "groups": len(groups.intersection(row.get("source_group") for row in seen)), "folders": len(directories)}


class ComparisonJobs:
    def __init__(self, store, prediction_jobs):
        self.store = store
        self.prediction_jobs = prediction_jobs
        self.directory = store.data / "comparisons"
        self.lock = threading.Lock()
        self.active = None
        self.stop = None
        self.thread = None
        for path in self.directory.glob("*/job.json"):
            record = read_state(path)
            if record.get("status") in ACTIVE_COMPARISON:
                atomic_json(path, {**record, "status": "interrupted", "finished_at": now()})

    def get(self, project_id, run_id):
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise Problem("模型对比任务不存在", 404)
        record = read_state(self.directory / run_id / "job.json")
        if record.get("id") != run_id or record.get("project") != project_id:
            raise Problem("模型对比任务不存在", 404)
        return record

    def list(self, project_id):
        self.store.project(project_id)
        records = [record for path in self.directory.glob("*/job.json")
                   if (record := read_state(path)).get("project") == project_id]
        with self.lock:
            for record in records:
                run_id = record["id"]
                if self.active == run_id and self.stop.is_set():
                    record["status"] = "cancelling"
                record["progress"] = read_state(self.directory / run_id / "progress.json")
        return sorted(records, key=lambda row: row["created_at"], reverse=True)[:20]

    def start(self, project_id, payload, *, task_id=None):
        if self.store.project(project_id).task_type != "detect":
            raise Problem("当前逐框模型对比适用于目标检测项目", 422)
        directory, snapshot = get_blindset(self.store, project_id, payload.get("set_id"))
        options = comparison_options(payload)
        weights = {name: weight_path(payload.get(f"{name}_path")) for name in ("old", "new")}
        if weights["old"] == weights["new"]:
            raise Problem("请选择两个不同版本的权重", 422)
        if not has_training_dependencies():
            raise Problem("当前 Python 缺少 PyTorch 或 Ultralytics", 422)
        self.prediction_jobs.reserve_model_slot()
        try:
            run_id = task_id or uuid.uuid4().hex
            run = self.directory / run_id
            record = {"id": run_id, "project": project_id, "set_id": snapshot["id"], "set_name": snapshot["name"],
                      "images": snapshot["images"], "status": "preparing", "created_at": now(), "options": options,
                      "old_name": weights["old"].name, "new_name": weights["new"].name}
            with self.lock:
                atomic_json(run / "job.json", record)
                self.active = run_id
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.run, args=(run, record, directory, snapshot, weights, self.stop), daemon=True)
                self.thread.start()
        except Exception:
            self.prediction_jobs.release_model_slot()
            raise
        return record

    def cancel(self, project_id, run_id):
        with self.lock:
            record = self.get(project_id, run_id)
            if self.active == run_id:
                self.stop.set()
        return record

    def shutdown(self):
        with self.lock:
            thread = self.thread
            if self.stop:
                self.stop.set()
        if thread:
            thread.join(timeout=6)

    def report(self, project_id, run_id):
        record = self.get(project_id, run_id)
        path = self.directory / run_id / "report.json"
        if record["status"] != "completed" or not path.is_file():
            raise Problem("对比报告尚不可用", 404)
        return path

    def details(self, project_id, run_id):
        record = self.get(project_id, run_id)
        path = self.directory / run_id / DETAILS_FILE
        if record["status"] != "completed" or not path.is_file():
            raise Problem("逐图记录尚不可用。早期记录可重新对比生成明细。", 404)
        return record, path.resolve()

    def run(self, run, record, directory, snapshot, weights, stop):
        result = {"status": "failed"}
        try:
            manifest = verified_manifest(directory, snapshot)
            models = {}
            with tempfile.TemporaryDirectory(dir=run, prefix=".weights-") as temporary:
                inputs = Path(temporary)
                for name, path in weights.items():
                    if stop.is_set():
                        raise LocalJobCancelled()
                    copied = inputs / f"{name}.pt"
                    shutil.copyfile(path, copied)
                    models[name] = {"filename": path.name, "sha256": digest(copied),
                                    "recorded_overlap": recorded_overlap(self.store, record["project"], path, snapshot, manifest)}
                if models["old"]["sha256"] == models["new"]["sha256"]:
                    raise Problem("两份权重内容相同，请选择两个版本", 422)
                if stop.is_set():
                    raise LocalJobCancelled()
                atomic_json(run / "compare.json", {"models": models, "options": record["options"],
                                                  "label_aliases": copy.deepcopy(self.store.project(record["project"]).label_aliases)})
                atomic_json(run / "job.json", {**record, "status": "running"})
                self.prediction_jobs.inference.release_cached_model()
                run_local_worker(run, stop, "labelpilot.evaluation_worker", directory, inputs)
                if stop.is_set():
                    raise LocalJobCancelled()
                if not (run / "report.json").is_file():
                    raise Problem("模型进程退出时未生成对比报告")
            result["status"] = "completed"
        except LocalJobCancelled:
            result["status"] = "cancelled"
        except Exception as exc:
            result["error"] = str(getattr(exc, "message", exc))[:500]
        finally:
            with self.lock:
                try:
                    atomic_json(run / "job.json", {**record, **result, "finished_at": now()})
                finally:
                    self.active = self.stop = self.thread = None
                    self.prediction_jobs.release_model_slot()
