import copy
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import replace
from pathlib import Path

from .blindsets import reserve_blind_images
from .exporting import DatasetFolder, SPLIT_UNITS, describe_export, plan_splits, prepare_export, write_members
from .store import Problem, atomic_json, digest, now, read_json


ACTIVE_TRAINING = ("preparing", "running")
BEST_WEIGHTS = Path("run/weights/best.pt")


class LocalJobCancelled(Exception):
    pass


def training_options(project, payload):
    unit = payload.get("unit", "source_group" if project.summary()["has_source_groups"] else "image")
    device = payload.get("device", "auto")
    mode = payload.get("mode", "scratch")
    batch = payload.get("batch", "")
    if unit not in SPLIT_UNITS or device not in ("auto", "cpu", "cuda") or mode not in ("scratch", "fine_tune"):
        raise Problem("无效的划分方式、训练设备或起始模型", 422)
    if not isinstance(batch, str):
        raise Problem("批次必须为文本", 422)
    options = {"unit": unit, "device": device, "mode": mode, "batch": batch}
    for key, label, default, minimum, maximum in (
        ("epochs", "训练轮数", 30, 1, 300), ("image_size", "训练尺寸", 640, 64, 1280),
        ("batch_size", "训练批大小", 4, 1, 64), ("seed", "随机种子", 42, 0, 2**31 - 1),
    ):
        value = payload.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise Problem(f"{label}必须为 {minimum} 到 {maximum} 的整数", 422)
        options[key] = value
    if options["image_size"] % 32:
        raise Problem("训练尺寸须为 32 的倍数", 422)
    return options


def read_state(path):
    try:
        state = read_json(path)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def has_training_dependencies():
    return all(importlib.util.find_spec(module) for module in ("torch", "ultralytics"))


class TrainingJobs:
    def __init__(self, store, inference, prediction_jobs):
        self.store = store
        self.inference = inference
        self.prediction_jobs = prediction_jobs
        self.directory = store.data / "training"
        self.lock = threading.Lock()
        self.records = {}
        self.worker = None
        self.active = None
        self.stop = None
        for metadata in self.directory.glob("*/job.json"):
            record = read_state(metadata)
            run_id = metadata.parent.name
            if not re.fullmatch(r"[0-9a-f]{32}", run_id) or record.get("id") != run_id:
                continue
            if record.get("status") in ACTIVE_TRAINING:
                record.update(status="interrupted", finished_at=now())
                atomic_json(metadata, record)
            self.records[run_id] = record

    def save_record(self, record):
        atomic_json(self.directory / record["id"] / "job.json", record)
        self.records[record["id"]] = record

    def view(self, record):
        run = self.directory / record["id"]
        result = copy.deepcopy(record)
        result["progress"] = read_state(run / "progress.json")
        result["has_log"] = (run / "train.log").is_file()
        result["has_dataset"] = (run / "dataset" / "export_manifest.json").is_file() and bool(record.get("split_counts"))
        if record["id"] == self.active and self.stop.is_set():
            result["status"] = "cancelling"
        elif record["status"] == "completed" and (run / BEST_WEIGHTS).is_file():
            result["weights"] = str(run / BEST_WEIGHTS)
        return result

    def list(self, project_id):
        with self.lock:
            records = [record for record in self.records.values() if record["project"] == project_id]
            return [self.view(record) for record in sorted(records, key=lambda row: row["created_at"], reverse=True)[:12]]

    def start(self, project_id, payload):
        project = self.store.project(project_id)
        options = training_options(project, payload)
        if not has_training_dependencies():
            raise Problem("当前 Python 缺少 PyTorch 或 Ultralytics，请使用安装了 LabelPilot 依赖的环境", 422)
        self.prediction_jobs.reserve_model_slot()
        try:
            weights = None
            if options["mode"] == "fine_tune":
                weights = Path(self.store.project(project_id).config.get("model_path") or "")
                if weights.suffix.lower() != ".pt" or not weights.is_file():
                    raise Problem("请先在模型设置里选择可信的本地 .pt 权重，或选择从零训练", 422)
            run_id = uuid.uuid4().hex
            record = {"id": run_id, "project": project_id, "status": "preparing", "created_at": now(), "options": options}
            with self.lock:
                self.save_record(record)
                self.stop = threading.Event()
                self.active = run_id
                self.worker = threading.Thread(target=self.run, args=(record, weights, self.stop), daemon=True)
                self.worker.start()
                return self.view(record)
        except Exception:
            self.prediction_jobs.release_model_slot()
            raise

    def get_record(self, project_id, run_id):
        record = self.records.get(run_id)
        if not record or record["project"] != project_id:
            raise Problem("训练任务不存在", 404)
        return record

    def cancel(self, project_id, run_id):
        with self.lock:
            record = self.get_record(project_id, run_id)
            if run_id == self.active:
                self.stop.set()
            return self.view(record)

    def artifact(self, project_id, run_id, kind):
        files = {"weights": BEST_WEIGHTS, "log": Path("train.log"), "dataset": Path("dataset/export_manifest.json")}
        with self.lock:
            record = self.get_record(project_id, run_id)
            if kind not in files or (kind == "weights" and record["status"] != "completed"):
                raise Problem("训练产物尚不可用", 404)
            if kind == "dataset" and not record.get("split_counts"):
                raise Problem("训练数据还在准备中", 404)
            run = self.directory / run_id
            path = (run / files[kind]).resolve()
            if not path.is_relative_to(run.resolve()) or not path.is_file():
                raise Problem("训练产物尚不可用", 404)
            return path

    def shutdown(self):
        with self.lock:
            worker = self.worker
            if self.stop:
                self.stop.set()
        if worker:
            worker.join(timeout=6)

    def run(self, record, weights, stop):
        def checkpoint():
            if stop.is_set():
                raise LocalJobCancelled()

        run = self.directory / record["id"]
        options = record["options"]
        updates = {"status": "failed"}
        try:
            with self.store.lock:
                data_options = {"format": "yolo", "batch": options["batch"], "clean": True}
                plan = prepare_export(self.store, record["project"], data_options, include_documents=True, checkpoint=checkpoint)
                rows = reserve_blind_images(self.store, record["project"], plan.rows)
                updates["blind_out"] = len(plan.rows) - len(rows)
                if not rows:
                    raise Problem("这一批图片已留作盲测，请选择其他训练批次", 422)
                split = {"train": 80, "val": 20, "test": 0, "unit": options["unit"], "seed": options["seed"]}
                assignments, split_config, split_counts = plan_splits(rows, {"split": split})
                plan = replace(plan, rows=rows, assignments=assignments, split_config=split_config, split_counts=split_counts)
                preview = describe_export(self.store, plan)
                if preview["blocking_issue"]:
                    raise Problem(preview["blocking_issue"], 422)
                if preview["cross_split_sources"]:
                    raise Problem("训练集和验证集共享已知来源组。请选择按来源组划分", 422)
                for subset, name in (("train", "训练集"), ("val", "验证集")):
                    if not any(row["splits"][subset] for row in preview["class_counts"]):
                        raise Problem(f"清洗划分后的{name}缺少带框图片。请补充已审核样本或调整本次划分", 422)
                write_members(self.store, plan, DatasetFolder(run / "dataset"), {}, checkpoint=checkpoint)
                names = list(plan.project.names)
            checkpoint()
            if weights is not None:
                starting = run / "starting.pt"
                shutil.copyfile(weights, starting)
                updates["starting_sha256"] = digest(starting)
            atomic_json(run / "train.json", {"options": options, "classes": names})
            checkpoint()
            updates.update(split_counts=plan.split_counts, cleaned_out=len(plan.cleaning))
            with self.lock:
                self.save_record({**record, **updates, "status": "running"})
            self.inference.release_cached_model()
            self.run_worker(run, stop)
            checkpoint()
            if not (run / BEST_WEIGHTS).is_file():
                raise Problem("训练已退出，但没有生成 best.pt")
            updates["status"] = "completed"
        except LocalJobCancelled:
            updates["status"] = "cancelled"
        except Exception as exc:
            updates["error"] = str(getattr(exc, "message", exc))[:500]
        finally:
            with self.lock:
                try:
                    self.save_record({**record, **updates, "finished_at": now()})
                finally:
                    self.active = self.stop = self.worker = None
                    self.prediction_jobs.release_model_slot()

    def run_worker(self, run, stop):
        run_local_worker(run, stop, "labelpilot.training_worker")


def run_local_worker(run, stop, module, *arguments):
    command = [sys.executable, "-X", "utf8", "-m", module, str(run), str(os.getpid()), *map(str, arguments)]
    with (run / "train.log").open("wb") as log, subprocess.Popen(
        command, cwd=Path(__file__).resolve().parent.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        while process.poll() is None:
            if stop.wait(0.25):
                (run / "cancel.request").touch()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise LocalJobCancelled()
        if process.returncode != 0:
            result = read_state(run / "error.json")
            raise Problem(result.get("error", f"本地模型进程退出（代码 {process.returncode}），请查看日志"))
