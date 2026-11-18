import io
import os
import re
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .exporting import export_project
from .inference import Inference, Jobs
from .store import Problem, Store


ASSETS = Path(__file__).resolve().parent.parent / "static"


def create_app(home=None):
    home = Path(home or os.environ.get("LABELPILOT_HOME", ASSETS.parent)).resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "data" / "ultralytics"))
    store = Store(home)
    inference = Inference(store)
    jobs = Jobs(store, inference)
    app = FastAPI(title="LabelPilot", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.inference = inference
    app.state.jobs = jobs
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
        if request.url.path == "/" or request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Problem)
    async def problem_handler(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (ASSETS / "index.html").read_text(encoding="utf-8").replace("__SESSION_TOKEN__", app.state.token)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "app": "LabelPilot", "projects": len(store.projects)}

    @app.get("/api/system")
    def system():
        import sys
        import torch
        return {"python": sys.executable, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "data_directory": str(store.data)}

    @app.get("/api/projects")
    def projects():
        return {"projects": [project.summary() for project in store.projects.values()], "errors": store.project_errors}

    @app.post("/api/projects")
    def register(payload: dict):
        if not isinstance(payload.get("image_root"), str) or not payload["image_root"].strip():
            raise Problem("请填写图片目录", 422)
        return store.register(payload)

    @app.post("/api/projects/{project_id}/model")
    def model(project_id: str, payload: dict):
        if not isinstance(payload.get("path"), str) or not payload["path"].strip():
            raise Problem("请填写权重路径", 422)
        return store.update_model(project_id, payload["path"])

    @app.post("/api/projects/{project_id}/labels")
    def add_label(project_id: str, payload: dict):
        return store.add_label(project_id, payload.get("name"))

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
