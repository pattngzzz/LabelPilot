import copy
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from .exporting import DatasetFolder, validated_export_plan, write_members
from .export_acceleration import AugmentationCache, ExportAcceleration, export_workers
from .incremental_export import record_metadata, safe_path
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
        workers = export_workers(options.get("workers", 0))
        reuse_cache = options.get("reuse_cache", True)
        if type(reuse_cache) is not bool:
            raise Problem("增强缓存开关必须为布尔值", 422)
        with self.lock:
            if self.closed:
                raise Problem("服务正在关闭，请稍后重试", 503)
            if self.worker and self.worker.is_alive():
                raise Problem("已有数据集正在导出，请等待完成后再开始", 409)
            job_id = uuid.uuid4().hex
            record = {"id": job_id, "project": project_id, "status": "preparing",
                      "destination": str(destination), "completed": 0, "total": 0, "percent": 0,
                      "current_image": "", "created_at": now(), "updated_at": now(), "workers": workers,
                      "cache_enabled": reuse_cache, "cache_hits": 0, "cache_misses": 0, "cache_errors": 0,
                      "elapsed_seconds": 0, "images_per_second": 0, "remaining_seconds": None}
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
        cache = None
        target_lock = None
        incremental = None
        cleanup_warning = None
        started = time.monotonic()
        writing_started = None

        def checkpoint():
            if self.stop.is_set():
                raise ExportCancelled()

        def progress(completed, total, image):
            elapsed = max(0.001, time.monotonic() - writing_started)
            speed = completed / elapsed
            self.update(job_id, completed=completed, total=total, percent=min(99, int(completed * 100 / total)),
                        current_image=image, elapsed_seconds=round(time.monotonic() - started, 1),
                        images_per_second=round(speed, 2), remaining_seconds=round((total - completed) / speed),
                        **(cache.stats() if cache else {}))

        try:
            checkpoint()
            if options.get("incremental") and (destination / "export_manifest.json").exists():
                target_lock = safe_path(destination, ".labelpilot-export.lock")
                try:
                    with target_lock.open("x", encoding="utf-8") as lock_file:
                        lock_file.write(job_id)
                except FileExistsError:
                    target_lock = None
                    raise Problem("该数据集已有增量导出锁；请等待其他任务完成。若上次异常退出，请先检查恢复目录再移除锁", 409)
            snapshot = ExportSnapshot(self.store, project_id)
            plan, fingerprints = validated_export_plan(snapshot, project_id, options, checkpoint=checkpoint)
            incremental = plan.incremental
            checkpoint()
            destination = directory(str(destination))
            safe_project = re.sub(r"[^\w-]", "_", project_id)
            filename = f"{safe_project}_{plan.format_name}_{job_id}"
            output = incremental.root if incremental else destination / filename
            staging_parent = destination.parent if incremental else destination
            staging = staging_parent / (f".lp-{job_id}.part" if incremental else f".{filename}.part")
            if (not incremental and output.exists()) or staging.exists():
                raise Problem("导出目录已存在，请重新导出，已有文件不会被覆盖", 409)
            self.update(job_id, status="writing", total=len(plan.rows) + plan.augmented_images,
                        incremental=incremental.summary() if incremental else None)
            dataset = DatasetFolder(staging)
            temporary = staging
            if options.get("reuse_cache", True) and plan.augmentation:
                cache = AugmentationCache(self.store.data / "export_cache")
            writing_started = time.monotonic()
            with ExportAcceleration(export_workers(options.get("workers", 0)), cache, checkpoint) as acceleration:
                manifest = write_members(snapshot, plan, dataset, fingerprints, checkpoint=checkpoint,
                                         progress=progress, verify=True, acceleration=acceleration)
            checkpoint()
            self.update(job_id, status="finalizing")
            if incremental:
                manifest = incremental.merge(plan, dataset, manifest)
                record_metadata(dataset, manifest)
                incremental.publish(dataset, checkpoint)
                try:
                    if temporary.is_symlink() or temporary.resolve().parent != incremental.root.parent:
                        raise OSError("临时目录路径已改变，未自动清理")
                    shutil.rmtree(temporary)
                except OSError:
                    cleanup_warning = f"数据已完成合并，但临时备份未能清理，请检查后移除：{temporary}"
                temporary = None
            else:
                record_metadata(dataset, manifest)
                if output.exists():
                    raise Problem("目标目录已存在，已有文件不会被覆盖", 409)
                temporary.rename(output)
                temporary = None
            result = {"directory": str(output), "images": len(manifest["items"]),
                      "original_images": manifest["original_images"], "augmented_images": manifest["augmented_images"],
                      "split_counts": manifest["split_counts"], "source_split_counts": manifest["source_split_counts"],
                      "incremental": incremental.summary() if incremental else None,
                      "cleanup_warning": cleanup_warning,
                       "integrity_check": manifest.get("integrity_check"), "acceleration": manifest.get("acceleration")}
            self.update(job_id, status="completed", percent=100, result=result, current_image="", remaining_seconds=0,
                        elapsed_seconds=round(time.monotonic() - started, 1), **(cache.stats() if cache else {}))
        except Exception as exc:
            cleanup_error = None
            if incremental and getattr(incremental, "recovery_required", False):
                cleanup_error = f"请保留恢复目录：{temporary} 和目标锁文件，不要删除。"
                temporary = None
                target_lock = None
            if temporary is not None and temporary.exists():
                try:
                    expected_parent = incremental.root.parent if incremental else destination
                    if temporary.is_symlink() or temporary.resolve().parent != expected_parent:
                        raise OSError("临时目录路径已改变，未自动清理")
                    shutil.rmtree(temporary)
                except OSError:
                    cleanup_error = f"未能清理临时目录：{temporary}，请检查后手动移除。"
            message = ("服务关闭，导出已停止" if isinstance(exc, ExportCancelled) else
                       exc.message if isinstance(exc, Problem) else f"导出失败，请检查目标磁盘空间、写入权限及源文件：{exc}")
            self.update(job_id, status="cancelled" if isinstance(exc, ExportCancelled) else "failed",
                        error=message + (f" {cleanup_error}" if cleanup_error else ""))
        finally:
            if target_lock is not None:
                target_lock.unlink(missing_ok=True)
            if snapshot is not None:
                snapshot.connection.close()
