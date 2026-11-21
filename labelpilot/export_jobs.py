import copy
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from .exporting import DatasetFolder, validated_export_plan, write_members
from .model_files import directory
from .store import Problem, Store, now


class ExportSnapshot(Store):
    def __init__(self, store, project_id):
        self.__dict__.update(store.__dict__)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        try:
            with store.lock:
                self.projects = {project_id: copy.deepcopy(store.project(project_id))}
                with store.connect() as connection:
                    connection.backup(self.connection)
            self.connection.execute("PRAGMA query_only=ON")
        except Exception:
            self.connection.close()
            raise

    @contextmanager
    def connect(self):
        yield self.connection


class ExportCancelled(Exception):
    pass


class ExportJobs:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.records = {}
        self.worker = None
        self.stop = threading.Event()
        self.closed = False

    def list(self, project_id):
        self.store.project(project_id)
        with self.lock:
            return [copy.deepcopy(record) for record in reversed(list(self.records.values()))
                    if record["project"] == project_id]

    def get(self, project_id, job_id):
        self.store.project(project_id)
        with self.lock:
            record = self.records.get(job_id)
            if not record or record["project"] != project_id:
                raise Problem("导出任务不存在，服务可能已重启，请重新导出", 404)
            return copy.deepcopy(record)

    def update(self, job_id, **values):
        with self.lock:
            self.records[job_id].update(values, updated_at=now())

    def start(self, project_id, options):
        self.store.project(project_id)
        destination = directory(options.get("destination"))
        with self.lock:
            if self.closed:
                raise Problem("服务正在关闭，请稍后重试", 503)
            if self.worker and self.worker.is_alive():
                raise Problem("已有数据集正在导出，请等待完成后再开始", 409)
            job_id = uuid.uuid4().hex
            record = {"id": job_id, "project": project_id, "status": "preparing",
                      "destination": str(destination), "completed": 0, "total": 0, "percent": 0,
                      "current_image": "", "created_at": now(), "updated_at": now()}
            self.records[job_id] = record
            while len(self.records) > 50:
                del self.records[next(iter(self.records))]
            self.stop.clear()
            self.worker = threading.Thread(target=self.run, args=(job_id, project_id, copy.deepcopy(options), destination),
                                           name="labelpilot-export", daemon=True)
            self.worker.start()
            return copy.deepcopy(record)

    def shutdown(self):
        with self.lock:
            self.closed = True
            self.stop.set()
            worker = self.worker
        if worker:
            worker.join()

    def run(self, job_id, project_id, options, destination):
        temporary = None
        snapshot = None

        def checkpoint():
            if self.stop.is_set():
                raise ExportCancelled()

        def progress(completed, total, image):
            self.update(job_id, completed=completed, total=total, percent=min(99, int(completed * 100 / total)),
                        current_image=image)

        try:
            checkpoint()
            snapshot = ExportSnapshot(self.store, project_id)
            plan, fingerprints = validated_export_plan(snapshot, project_id, options, checkpoint=checkpoint)
            checkpoint()
            destination = directory(str(destination))
            safe_project = re.sub(r"[^\w-]", "_", project_id)
            filename = f"{safe_project}_{plan.format_name}_{job_id}"
            output = destination / filename
            staging = destination / f".{filename}.part"
            if output.exists() or staging.exists():
                raise Problem("导出目录已存在，请重新导出，已有文件不会被覆盖", 409)
            self.update(job_id, status="writing", total=len(plan.rows) + plan.augmented_images)
            dataset = DatasetFolder(staging)
            temporary = staging
            manifest = write_members(snapshot, plan, dataset, fingerprints, checkpoint=checkpoint, progress=progress, verify=True)
            checkpoint()
            self.update(job_id, status="finalizing")
            if output.exists():
                raise Problem("目标目录已存在，已有文件不会被覆盖", 409)
            temporary.rename(output)
            temporary = None
            result = {"directory": str(output), "images": len(manifest["items"]),
                      "original_images": len(plan.rows), "augmented_images": plan.augmented_images,
                      "split_counts": manifest["split_counts"], "source_split_counts": plan.split_counts,
                      "integrity_check": manifest.get("integrity_check")}
            self.update(job_id, status="completed", percent=100, result=result, current_image="")
        except Exception as exc:
            cleanup_error = None
            if temporary is not None and temporary.exists():
                try:
                    if temporary.is_symlink() or temporary.resolve().parent != destination:
                        raise OSError("临时目录路径已改变，未自动清理")
                    shutil.rmtree(temporary)
                except OSError:
                    cleanup_error = f"未能清理临时目录：{temporary}，请检查后手动移除。"
            message = ("服务关闭，导出已停止" if isinstance(exc, ExportCancelled) else
                       exc.message if isinstance(exc, Problem) else f"导出失败，请检查目标磁盘空间、写入权限及源文件：{exc}")
            self.update(job_id, status="cancelled" if isinstance(exc, ExportCancelled) else "failed",
                        error=message + (f" {cleanup_error}" if cleanup_error else ""))
        finally:
            if snapshot is not None:
                snapshot.connection.close()
