import hashlib
import io
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .annotation_tools import inherit_next, interpolate_frames, prompt_box
from .augmentation import augmentation_presets
from .automation import Automation
from .blindsets import freeze_blindset, frozen_image_bytes, frozen_image_path, get_blindset, list_blindsets
from .comparison_details import review_frame, review_frames
from .curation import similar_frames
from .evaluation import ComparisonJobs
from .exporting import export_project, preview_export
from .inference import Inference, Jobs
from . import model_files, native_picker
from .store import IMAGE_SUFFIXES, Problem, Store
from .tracking import track_frames
from .training import TrainingJobs, has_training_dependencies


ASSETS = Path(__file__).resolve().parent.parent / "static"


def create_app(home=None):
    home = Path(home or os.environ.get("LABELPILOT_HOME", ASSETS.parent)).resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "data" / "ultralytics"))
    store = Store(home)
    inference = Inference(store)
    jobs = Jobs(store, inference)
    training = TrainingJobs(store, inference, jobs)
    comparisons = ComparisonJobs(store, jobs)
    automation = Automation(store, jobs, training, comparisons)

    @asynccontextmanager
    async def lifespan(app):
        automation.begin()
        try:
            yield
        finally:
            automation.shutdown()
            comparisons.shutdown()
            training.shutdown()

    app = FastAPI(title="LabelPilot", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = store
    app.state.inference = inference
    app.state.jobs = jobs
    app.state.training = training
    app.state.comparisons = comparisons
    app.state.automation = automation
    app.state.token = secrets.token_urlsafe(32)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def protect_local(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and (urlsplit(origin).netloc != request.headers.get("host") or urlsplit(origin).scheme not in ("http", "https")):
            return JSONResponse({"detail": "拒绝跨站请求"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not secrets.compare_digest(request.headers.get("x-labelpilot-token", ""), app.state.token):
                return JSONResponse({"detail": "本地会话已过期，请刷新页面"}, status_code=403)
            if int(request.headers.get("content-length", "0")) > 20 * 1024 * 1024:
                return JSONResponse({"detail": "请求过大"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/" or request.url.path.startswith(("/api", "/static/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Problem)
    async def problem_handler(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.get("/", response_class=HTMLResponse)
    def index():
        version = hashlib.sha256(b"".join((ASSETS / name).read_bytes() for name in
                                         ("index.html", "efficiency.js", "resume.js", "automation.js", "task-tools.js",
                                          "app.js", "style.css", "realityloop.png"))).hexdigest()[:16]
        return (ASSETS / "index.html").read_text(encoding="utf-8").replace(
            "__SESSION_TOKEN__", app.state.token).replace("__ASSET_VERSION__", version)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "app": "LabelPilot", "projects": len(store.projects)}

    @app.get("/api/system")
    def system():
        import sys
        try:
            import torch
        except (ImportError, OSError):
            torch = None
        cuda = bool(torch and torch.cuda.is_available())
        return {"python": sys.executable, "torch": str(torch.__version__) if torch else None, "cuda": cuda,
                "device": torch.cuda.get_device_name(0) if cuda else "CPU", "data_directory": str(store.data),
                "training_available": bool(torch and has_training_dependencies())}

    @app.get("/api/projects")
    def projects():
        return {"projects": [project.summary() for project in store.projects.values()], "errors": store.project_errors}

    @app.post("/api/projects")
    def register(payload: dict):
        if not isinstance(payload.get("image_root"), str) or not payload["image_root"].strip():
            raise Problem("请填写图片目录", 422)
        return store.register(payload)

    @app.post("/api/images/browse")
    def browse_images(payload: dict):
        path = payload.get("path")
        return model_files.browse(str(Path.home()) if path is None else path, IMAGE_SUFFIXES)

    @app.post("/api/images/select")
    def select_images(payload: dict):
        return native_picker.select_images(payload.get("mode"), payload.get("path"))

    @app.post("/api/projects/{project_id}/model")
    def model(project_id: str, payload: dict):
        if not isinstance(payload.get("path"), str) or not payload["path"].strip():
            raise Problem("请填写权重路径", 422)
        with jobs.lock:
            active = any(job["project"] == project_id and job["status"] in ("queued", "running") for job in jobs.records.values())
            if active or not inference.lock.acquire(blocking=False):
                raise Problem("模型任务运行期间不能切换权重，请等待或取消任务", 409)
            try:
                return store.update_model(project_id, payload["path"])
            finally:
                inference.lock.release()

    @app.post("/api/projects/{project_id}/model/browse")
    def browse_model(project_id: str, payload: dict):
        project = store.project(project_id)
        path = payload.get("path")
        if path is None:
            configured = Path(project.config.get("model_path") or home)
            path = str(configured.parent if configured.suffix.lower() == ".pt" else configured)
            if not Path(path).is_dir():
                path = str(Path.home())
        return model_files.browse(path)

    @app.post("/api/projects/{project_id}/model/search")
    def search_models(project_id: str, payload: dict):
        store.project(project_id)
        return model_files.search(payload.get("path"))

    @app.post("/api/projects/{project_id}/labels")
    def add_label(project_id: str, payload: dict):
        return store.add_label(project_id, payload.get("name"))

    @app.post("/api/projects/{project_id}/labels/import")
    def import_labels(project_id: str, payload: dict):
        return store.import_labels(project_id, payload.get("filename"), payload.get("content"))

    @app.post("/api/projects/{project_id}/labels/delete")
    def delete_label(project_id: str, payload: dict):
        store.project(project_id)
        active = {"queued", "running", "waiting", "preparing", "cancelling"}
        if (inference.lock.locked() or
                any(job["status"] in active for job in jobs.list(project_id)) or
                any(run["status"] in active for run in training.list(project_id)) or
                any(run["status"] in active for run in comparisons.list(project_id)) or
                any(run["status"] in active for run in automation.view(project_id)["runs"])):
            raise Problem("项目有运行中的模型或自动任务，请等待任务完成后再删除标签", 409)
        jobs.reserve_model_slot()
        try:
            return store.delete_label(project_id, payload.get("name"), payload.get("expected_classes"))
        finally:
            jobs.release_model_slot()

    @app.get("/api/projects/{project_id}/items")
    def items(project_id: str, query: str = "", batch: str = "", status: str = "",
              offset: int = Query(0, ge=0), limit: int = Query(80, ge=1, le=10000)):
        return store.listing(project_id, query, batch, status, offset, limit)

    @app.post("/api/projects/{project_id}/rescan")
    def rescan(project_id: str):
        if inference.lock.locked() or any(job["status"] in ("queued", "running") for job in jobs.list(project_id)):
            raise Problem("模型任务运行期间不能重新扫描，请等待或取消任务", 409)
        return store.rescan(project_id)

    @app.get("/api/projects/{project_id}/items/{item_id}")
    def detail(project_id: str, item_id: str):
        return store.detail(project_id, item_id)

    @app.put("/api/projects/{project_id}/items/{item_id}")
    def save(project_id: str, item_id: str, payload: dict):
        return store.save(project_id, item_id, payload)

    @app.get("/api/projects/{project_id}/items/{item_id}/history")
    def history(project_id: str, item_id: str):
        return store.history(project_id, item_id)

    @app.post("/api/projects/{project_id}/items/{item_id}/restore")
    def restore(project_id: str, item_id: str, payload: dict):
        return store.restore(project_id, item_id, payload.get("revision"), payload.get("expected_version"))

    @app.get("/api/projects/{project_id}/items/{item_id}/image")
    def image(project_id: str, item_id: str, thumbnail: bool = False):
        project = store.project(project_id)
        item = project.item(item_id)
        path = Path(item["path"])
        if not path.is_file() or not path.resolve().is_relative_to(project.root):
            raise Problem("原图路径已改变", 409)
        if not thumbnail:
            return FileResponse(path)
        with Image.open(path) as original:
            original.thumbnail((180, 120))
            stream = io.BytesIO()
            original.convert("RGB").save(stream, "JPEG", quality=72)
        return Response(stream.getvalue(), media_type="image/jpeg")

    @app.post("/api/projects/{project_id}/items/{item_id}/predict")
    def predict(project_id: str, item_id: str, payload: dict):
        return inference.predict(project_id, item_id, payload)

    @app.post("/api/projects/{project_id}/items/{item_id}/prompt-box")
    def propose_box(project_id: str, item_id: str, payload: dict):
        return prompt_box(store, project_id, item_id, payload)

    @app.post("/api/projects/{project_id}/sequence/inherit")
    def sequence_inherit(project_id: str, payload: dict):
        return inherit_next(store, project_id, payload)

    @app.post("/api/projects/{project_id}/sequence/preview")
    def sequence_preview(project_id: str, payload: dict):
        return interpolate_frames(store, project_id, payload)

    @app.post("/api/projects/{project_id}/sequence/apply")
    def sequence_apply(project_id: str, payload: dict):
        return interpolate_frames(store, project_id, payload, apply=True)

    @app.post("/api/projects/{project_id}/tracking/preview")
    def tracking_preview(project_id: str, payload: dict):
        return track_frames(store, project_id, payload)

    @app.post("/api/projects/{project_id}/tracking/apply")
    def tracking_apply(project_id: str, payload: dict):
        return track_frames(store, project_id, payload, apply=True)

    @app.get("/api/projects/{project_id}/automation")
    def automation_status(project_id: str):
        return automation.view(project_id)

    @app.put("/api/projects/{project_id}/automation")
    def automation_settings(project_id: str, payload: dict):
        return automation.configure(project_id, payload)

    @app.post("/api/projects/{project_id}/automation/runs", status_code=202)
    def automation_run(project_id: str, payload: dict):
        return automation.enqueue(project_id, payload.get("kind"), payload.get("batch", ""), configuration=payload.get("recipe"))

    @app.post("/api/projects/{project_id}/automation/runs/{run_id}/retry", status_code=202)
    def automation_retry(project_id: str, run_id: str):
        return automation.retry(project_id, run_id)

    @app.post("/api/projects/{project_id}/automation/runs/{run_id}/activate")
    def automation_activate(project_id: str, run_id: str):
        return automation.activate(project_id, run_id)

    @app.post("/api/projects/{project_id}/similarity/scan")
    def similarity_scan(project_id: str, payload: dict):
        return similar_frames(store, project_id, payload.get("batch", ""), payload.get("level", "balanced"))

    @app.post("/api/projects/{project_id}/similarity/selection")
    def similarity_selection(project_id: str, payload: dict):
        return store.save_similarity_selection(project_id, payload.get("batch", ""), payload.get("excluded"))

    @app.post("/api/projects/{project_id}/audits", status_code=202)
    def start_audit(project_id: str, payload: dict):
        return jobs.start(project_id, payload, audit=True)

    @app.get("/api/projects/{project_id}/items/{item_id}/audit")
    def item_audit(project_id: str, item_id: str):
        return store.audit_detail(project_id, item_id)

    @app.post("/api/projects/{project_id}/items/{item_id}/audit/dismiss")
    def dismiss_audit(project_id: str, item_id: str, payload: dict):
        return store.dismiss_audit(project_id, item_id, payload.get("id"))

    @app.get("/api/projects/{project_id}/jobs")
    def list_jobs(project_id: str):
        store.project(project_id)
        return jobs.list(project_id)

    @app.post("/api/projects/{project_id}/jobs")
    def start_job(project_id: str, payload: dict):
        return jobs.start(project_id, payload)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    @app.get("/api/projects/{project_id}/trainings")
    def list_trainings(project_id: str):
        store.project(project_id)
        return training.list(project_id)

    @app.post("/api/projects/{project_id}/trainings", status_code=202)
    def start_training(project_id: str, payload: dict):
        return training.start(project_id, payload)

    @app.post("/api/projects/{project_id}/trainings/{run_id}/cancel")
    def cancel_training(project_id: str, run_id: str):
        return training.cancel(project_id, run_id)

    @app.get("/api/projects/{project_id}/trainings/{run_id}/{kind}")
    def training_artifact(project_id: str, run_id: str, kind: str):
        path = training.artifact(project_id, run_id, kind)
        if kind == "log":
            with path.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 65536))
                return PlainTextResponse(stream.read(65536).decode("utf-8", errors="replace"))
        name = f"labelpilot-{run_id[:8]}-best.pt" if kind == "weights" else "export_manifest.json"
        return FileResponse(path, filename=name)

    def snapshot_summary(record):
        return {key: record[key] for key in ("id", "name", "created_at", "batch", "percent", "unit", "images",
                                             "objects", "confirmed_empty", "cleaned_out", "manifest_sha256")}

    @app.get("/api/projects/{project_id}/blindsets")
    def blindsets(project_id: str):
        return [snapshot_summary(record) for record in list_blindsets(store, project_id)]

    @app.post("/api/projects/{project_id}/blindsets", status_code=201)
    def create_blindset(project_id: str, payload: dict):
        return snapshot_summary(freeze_blindset(store, project_id, payload))

    @app.get("/api/projects/{project_id}/comparison-models")
    def comparison_models(project_id: str):
        project = store.project(project_id)
        current = project.config.get("model_path")
        paths = {}
        if current and Path(current).is_file():
            paths[current] = "当前项目模型"
        for run in training.list(project_id):
            if run.get("weights"):
                paths.setdefault(run["weights"], f"本地训练 {run['id'][:8]} · {run['created_at'][:10]}")
        return [{"path": path, "label": label} for path, label in paths.items()]

    @app.get("/api/projects/{project_id}/comparisons")
    def list_comparisons(project_id: str):
        return comparisons.list(project_id)

    @app.post("/api/projects/{project_id}/comparisons", status_code=202)
    def start_comparison(project_id: str, payload: dict):
        return comparisons.start(project_id, payload)

    @app.post("/api/projects/{project_id}/comparisons/{run_id}/cancel")
    def cancel_comparison(project_id: str, run_id: str):
        return comparisons.cancel(project_id, run_id)

    @app.get("/api/projects/{project_id}/comparisons/{run_id}/report")
    def comparison_report(project_id: str, run_id: str):
        path = comparisons.report(project_id, run_id)
        return FileResponse(path, filename=f"labelpilot-compare-{run_id[:8]}.json")

    @app.get("/api/projects/{project_id}/comparisons/{run_id}/frames")
    def comparison_frames(project_id: str, run_id: str, label: str, kind: str,
                          offset: int = Query(0, ge=0, le=10000), limit: int = Query(30, ge=1, le=100)):
        _, database = comparisons.details(project_id, run_id)
        return review_frames(database, label, kind, offset, limit)

    @app.get("/api/projects/{project_id}/comparisons/{run_id}/frames/{frame_id}")
    def comparison_frame(project_id: str, run_id: str, frame_id: str):
        _, database = comparisons.details(project_id, run_id)
        return review_frame(database, frame_id)

    @app.get("/api/projects/{project_id}/comparisons/{run_id}/frames/{frame_id}/image")
    def comparison_frame_image(project_id: str, run_id: str, frame_id: str):
        record, database = comparisons.details(project_id, run_id)
        item = review_frame(database, frame_id, image=True)
        directory, _ = get_blindset(store, project_id, record["set_id"])
        path = frozen_image_path(directory, item)
        contents = frozen_image_bytes(path, item)
        with Image.open(io.BytesIO(contents)) as original:
            media = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(original.format)
            if media:
                return Response(contents, media_type=media)
            output = io.BytesIO()
            original.convert("RGB").save(output, "PNG")
        return Response(output.getvalue(), media_type="image/png")

    @app.post("/api/projects/{project_id}/export/preview")
    def export_preview(project_id: str, payload: dict):
        return preview_export(store, project_id, payload)

    @app.get("/api/augmentation-presets")
    def export_augmentation_presets():
        return augmentation_presets()

    @app.post("/api/projects/{project_id}/export")
    def export(project_id: str, payload: dict):
        return export_project(store, project_id, payload)

    @app.get("/api/exports/{export_id}")
    def download(export_id: str):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", export_id):
            raise Problem("无效导出编号", 422)
        path = store.home / "exports" / f"{export_id}.zip"
        if not path.is_file():
            raise Problem("导出文件不存在", 404)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    app.mount("/static", StaticFiles(directory=ASSETS), name="static")
    return app
