import copy
import threading
import time
import uuid

from .exporting import preview_export
from .incremental_export import ExportFileCache
from .store import Problem, now


class PreviewCancelled(Exception):
    pass


class ExportPreviews:
    def __init__(self, store):
        self.store = store
        if not hasattr(store, "export_file_cache"):
            store.export_file_cache = ExportFileCache()
        self.lock = threading.RLock()
        self.records = {}
        self.pending = None
        self.active = None
        self.worker = None
        self.closed = False

    def get(self, project_id, job_id):
        self.store.project(project_id)
        with self.lock:
            record = self.records.get(job_id)
            if not record or record["project"] != project_id:
                raise Problem("预检任务不存在，服务可能已重启，请刷新预检", 404)
            result = copy.deepcopy(record)
            for task in (self.active, self.pending):
                if task and task["id"] == job_id:
                    result["elapsed_seconds"] = round(time.monotonic() - task["started"], 1)
            return result

    def start(self, project_id, options):
        self.store.project(project_id)
        with self.lock:
            if self.closed:
                raise Problem("服务正在关闭，请稍后重试", 503)
            for task in (self.pending, self.active):
                if (task and not task["stop"].is_set() and task["project"] == project_id
                        and self.records[task["id"]]["status"] in ("queued", "running")
                        and task["options"] == options):
                    return self.get(project_id, task["id"])
            for task in (self.pending, self.active):
                if task:
                    self.cancel(task["project"], task["id"])
            job_id = uuid.uuid4().hex
            record = {"id": job_id, "project": project_id, "status": "queued", "stage": "等待后台预检",
                      "completed": 0, "total": 0, "percent": None, "current_image": "",
                      "elapsed_seconds": 0, "created_at": now(), "updated_at": now()}
            self.records[job_id] = record
            self.pending = {"id": job_id, "project": project_id, "options": copy.deepcopy(options),
                            "stop": threading.Event(), "started": time.monotonic()}
            for previous in list(self.records):
                if len(self.records) <= 32:
                    break
                if (self.records[previous]["status"] in ("completed", "failed", "cancelled")
                        and (not self.active or previous != self.active["id"])):
                    del self.records[previous]
            if self.worker is None:
                self.worker = threading.Thread(target=self.run, name="labelpilot-export-preview", daemon=True)
                self.worker.start()
            return copy.deepcopy(record)

    def cancel(self, project_id, job_id):
        with self.lock:
            record = self.get(project_id, job_id)
            if record["status"] in ("completed", "failed", "cancelled"):
                return record
            for task in (self.pending, self.active):
                if task and task["id"] == job_id:
                    task["stop"].set()
                    pending = task is self.pending
                    self.records[job_id].update(status="cancelled" if pending else "cancelling", updated_at=now())
                    if pending:
                        self.pending = None
            return self.get(project_id, job_id)

    def run(self):
        while True:
            with self.lock:
                task = self.pending
                self.pending = None
                self.active = task
                if task is None:
                    self.worker = None
                    return
            self.run_one(task)
            with self.lock:
                self.active = None

    def run_one(self, task):
        job_id = task["id"]
        last_update = 0

        def checkpoint():
            if task["stop"].is_set():
                raise PreviewCancelled()

        def progress(stage, completed, total, image):
            nonlocal last_update
            checkpoint()
            current = time.monotonic()
            with self.lock:
                record = self.records[job_id]
                if record["stage"] == stage and completed < total and current - last_update < 0.15:
                    return
                record.update(stage=stage, completed=completed, total=total, current_image=image,
                              percent=int(completed * 100 / total) if total else None, updated_at=now())
            last_update = current

        try:
            checkpoint()
            with self.lock:
                checkpoint()
                self.records[job_id].update(status="running", updated_at=now())
            result = preview_export(self.store, task["project"], task["options"],
                                    checkpoint=checkpoint, progress=progress)
            checkpoint()
            with self.lock:
                checkpoint()
                self.records[job_id].update(status="completed", result=result, stage="预检完成", percent=100,
                                            updated_at=now())
        except Exception as exc:
            with self.lock:
                self.records[job_id].update(
                    status="cancelled" if isinstance(exc, PreviewCancelled) else "failed",
                    error="预检已停止" if isinstance(exc, PreviewCancelled) else
                    exc.message if isinstance(exc, Problem) else f"预检失败：{exc}",
                    error_status=exc.status if isinstance(exc, Problem) else 500, updated_at=now())
        finally:
            with self.lock:
                self.records[job_id]["elapsed_seconds"] = round(time.monotonic() - task["started"], 1)

    def shutdown(self):
        with self.lock:
            self.closed = True
            for task in (self.pending, self.active):
                if task:
                    self.cancel(task["project"], task["id"])
            worker = self.worker
        if worker:
            worker.join()
