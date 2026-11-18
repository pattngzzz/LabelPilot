import copy
import hashlib
import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


STATUSES = ("unstarted", "imported", "predicted", "draft", "reviewed", "skipped", "error")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = ["#a78bfa", "#38bdf8", "#fb923c", "#f472b6", "#34d399", "#facc15",
          "#60a5fa", "#e879f9", "#2dd4bf", "#f87171", "#a3e635", "#fbbf24", "#22d3ee"]


class Problem(Exception):
    def __init__(self, message, status=400):
        self.message = message
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_classes(config):
    if config.get("class_map_path"):
        path = Path(config["class_map_path"])
        if not path.is_file():
            raise Problem("类别文件不存在")
        if path.suffix.lower() == ".txt":
            names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            value = read_json(path)
            names = value.get("names", value) if isinstance(value, dict) else value
            if isinstance(names, dict):
                if sorted(int(key) for key in names) != list(range(len(names))):
                    raise Problem("类别 ID 必须从 0 连续排列")
                names = [names[str(index)] for index in range(len(names))]
    else:
        names = config.get("classes", [])
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise Problem("请提供有序类别列表，或 names 格式的 class_map.json / classes.txt")
    if len(set(names)) != len(names):
        raise Problem("类别名称重复")
    if any("\n" in name or "\r" in name for name in names):
        raise Problem("类别名称不能包含换行")
    return names


class Project:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.id = config["id"]
        self.root = Path(config["image_root"]).resolve()
        self.names = load_classes(config)
        self.items = []
        self.by_id = {}
        self.warnings = []
        self.scan()

    def source_map(self, directory):
        result = {}
        if not directory:
            return result
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise Problem(f"标注目录不存在：{directory}")
        for path in sorted(directory.rglob("*.json")):
            if path.stem not in self.photo_ids:
                continue
            if path.stem in result:
                raise Problem(f"标注文件同名冲突，不能自动选择：{path.stem}")
            result[path.stem] = path
        return result

    def scan(self):
        if not self.root.is_dir():
            raise Problem("图片目录不存在")
        manifest_path = self.config.get("manifest_path")
        if manifest_path:
            for task in read_json(manifest_path).get("tasks", []):
                if not task.get("package_image"):
                    continue
                path = (self.root / task["package_image"]).resolve()
                if not path.is_relative_to(self.root):
                    raise Problem("任务清单包含图片目录之外的路径")
                if path.is_file():
                    self.add_item(path, task)
        else:
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    self.add_item(path.resolve(), {})
        self.photo_ids = {item["photo_id"] for item in self.items}
        native = self.source_map(self.config.get("annotation_root"))
        predicted = self.source_map(self.config.get("preannotation_root"))
        counts = {}
        for item in self.items:
            counts[item["photo_id"]] = counts.get(item["photo_id"], 0) + 1
        for item in self.items:
            photo_id = item["photo_id"]
            source = native.get(photo_id) or predicted.get(photo_id)
            if source and counts[photo_id] > 1:
                item.update(status="error", error="同名图片对应的标注有歧义，请拆分项目或使用唯一图片名")
                continue
            if source:
                try:
                    document = read_json(source)
                    if not isinstance(document.get("shapes"), list):
                        raise ValueError("缺少 shapes 数组")
                    is_prediction = bool(document.get("preannotation")) or photo_id not in native
                    item.update(source=str(source), status="predicted" if is_prediction else "imported",
                                count=len(document["shapes"]), source_kind="model" if is_prediction else "human")
                except (ValueError, OSError, TypeError) as exc:
                    item.update(status="error", error=f"标注读取失败：{exc}")
        self.items.sort(key=lambda item: (item["batch"], item["photo_id"], item["relative"]))
        if not self.items:
            self.warnings.append("未找到图片；请确认图片目录或任务清单路径")
        self.by_id = {item["id"]: item for item in self.items}

    def add_item(self, path, task):
        if not path.is_relative_to(self.root):
            raise Problem("图片链接指向项目目录之外")
        relative = path.relative_to(self.root).as_posix()
        item_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
        if item_id in self.by_id:
            raise Problem(f"图片清单重复：{relative}")
        self.by_id[item_id] = True
        self.items.append({
            "id": item_id, "photo_id": task.get("photo_id", path.stem), "path": str(path),
            "relative": relative, "width": task.get("width"), "height": task.get("height"),
            "image_sha256": task.get("sha256"), "batch": task.get("batch") or path.parent.name,
            "source_group": task.get("source_group", "unassigned"),
            "source_id": task.get("source_id", ""), "note": task.get("visual_note", ""),
            "status": "unstarted", "count": 0, "source": None, "source_kind": None,
        })

    def item(self, item_id):
        try:
            return self.by_id[item_id]
        except KeyError:
            raise Problem("图片不存在", 404)

    def summary(self):
        return {"id": self.id, "name": self.config["name"], "image_root": str(self.root),
                "classes": [{"id": index, "name": name, "display": name,
                             "color": COLORS[index % len(COLORS)]} for index, name in enumerate(self.names)],
                "model_path": self.config.get("model_path", ""), "total": len(self.items),
                "batches": sorted({item["batch"] for item in self.items}), "warnings": self.warnings}


class Store:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.data = self.home / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.home / "projects.json"
        self.lock = threading.RLock()
        self.projects = {}
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS annotations (
                    project TEXT NOT NULL, item TEXT NOT NULL, version INTEGER NOT NULL,
                    status TEXT NOT NULL, document TEXT NOT NULL, updated TEXT NOT NULL,
                    PRIMARY KEY (project, item)
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    project TEXT NOT NULL, item TEXT NOT NULL, version INTEGER NOT NULL,
                    status TEXT NOT NULL, document TEXT NOT NULL, updated TEXT NOT NULL,
                    PRIMARY KEY (project, item, version)
                );
            """)
        self.configs = read_json(self.registry_path) if self.registry_path.exists() else []
        self.project_errors = []
        for config in self.configs:
            try:
                self.projects[config["id"]] = Project(config)
            except (Problem, OSError, ValueError, KeyError) as exc:
                self.project_errors.append({"name": config.get("name"), "error": str(getattr(exc, "message", exc))})

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.data / "workbench.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def project(self, project_id):
        if project_id not in self.projects:
            raise Problem("项目不存在", 404)
        return self.projects[project_id]

    def register(self, config):
        config = copy.deepcopy(config)
        config["id"] = uuid.uuid4().hex[:12]
        if not isinstance(config.get("name"), str) or not config["name"].strip():
            raise Problem("请填写项目名称")
        root = Path(config.get("image_root", "")).resolve()
        if root == self.home or root.is_relative_to(self.home) or self.home.is_relative_to(root):
            raise Problem("图片目录与工具目录必须独立，避免将缓存和导出重复导入")
        config["image_root"] = str(root)
        for field in ("annotation_root", "preannotation_root", "manifest_path", "class_map_path", "model_path"):
            if config.get(field):
                config[field] = str(Path(config[field]).resolve())
        project = Project(config)
        with self.lock:
            updated = self.configs + [config]
            atomic_json(self.registry_path, updated)
            self.configs = updated
            self.projects[project.id] = project
        return project.summary()

    def update_model(self, project_id, path):
        project = self.project(project_id)
        path = Path(path).resolve()
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise Problem("请选择可信的本地 Ultralytics .pt 检测模型")
        with self.lock:
            configs = copy.deepcopy(self.configs)
            next(config for config in configs if config["id"] == project_id)["model_path"] = str(path)
            atomic_json(self.registry_path, configs)
            self.configs = configs
            project.config["model_path"] = str(path)
        return project.summary()

    def rescan(self, project_id):
        with self.lock:
            project = self.project(project_id)
            replacement = Project(project.config)
            self.projects[project_id] = replacement
        return replacement.summary()

    def add_label(self, project_id, name):
        if not isinstance(name, str):
            raise Problem("标签名称必须是文本", 422)
        name = name.strip()
        if not name or len(name) > 160 or any(ord(character) < 32 for character in name):
            raise Problem("标签名称必须为 1 到 160 个字符，不能包含换行或控制字符", 422)
        with self.lock:
            project = self.project(project_id)
            if name in project.names:
                return project.summary()
            names = project.names + [name]
            configs = copy.deepcopy(self.configs)
            config = next(config for config in configs if config["id"] == project_id)
            if config.get("class_map_path"):
                config["imported_class_map_path"] = config.pop("class_map_path")
            config["classes"] = names
            atomic_json(self.registry_path, configs)
            self.configs = configs
            project.config = copy.deepcopy(config)
            project.names = names
        return project.summary()

    def overlay(self, project_id):
        with self.connect() as connection:
            rows = connection.execute("SELECT item, version, status, document, updated FROM annotations WHERE project=?", (project_id,)).fetchall()
        return {row["item"]: {"version": row["version"], "status": row["status"],
                               "count": len(json.loads(row["document"])["shapes"]), "updated": row["updated"]} for row in rows}

    def listing(self, project_id, query="", batch="", status="", offset=0, limit=80):
        project = self.project(project_id)
        overlays = self.overlay(project_id)
        counts = dict.fromkeys(STATUSES, 0)
        filtered = []
        for original in project.items:
            item = {key: value for key, value in original.items() if key not in ("path", "source")}
            item.update(overlays.get(item["id"], {}))
            counts[item["status"]] += 1
            if query.lower() not in (item["photo_id"] + " " + item["relative"]).lower():
                continue
            if batch and item["batch"] != batch:
                continue
            if status and item["status"] != status:
                continue
            filtered.append(item)
        return {"items": filtered[offset:offset + limit], "filtered": len(filtered),
                "total": len(project.items), "counts": counts, "offset": offset, "limit": limit}

    def image_info(self, project, item):
        path = Path(item["path"])
        if not path.is_file() or not path.resolve().is_relative_to(project.root):
            raise Problem("原图不存在或路径已改变", 409)
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274, 1) != 1:
                raise Problem("此图含 EXIF 旋转，需先生成方向归一化的独立数据副本，避免标注错位", 422)
        if item["width"] and (width, height) != (item["width"], item["height"]):
            raise Problem("原图尺寸与任务清单不符", 409)
        image_digest = digest(path)
        if item["image_sha256"] and image_digest != item["image_sha256"]:
            raise Problem("原图 SHA-256 与任务清单不符；已阻止编辑", 409)
        return width, height, image_digest

    def detail(self, project_id, item_id):
        project = self.project(project_id)
        item = project.item(item_id)
        if item.get("error"):
            raise Problem(item["error"], 422)
        width, height, image_digest = self.image_info(project, item)
        source_digest = digest(item["source"]) if item["source"] else None
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM annotations WHERE project=? AND item=?", (project_id, item_id)).fetchone()
        if row:
            document = json.loads(row["document"])
            version, status = row["version"], row["status"]
            recorded = document.get("labelpilot", {}).get("image_sha256")
            if recorded and recorded != image_digest:
                raise Problem("原图已在上次保存后改变；已阻止旧标注套用", 409)
        elif item["source"]:
            document = read_json(item["source"])
            version, status = 0, item["status"]
        else:
            document = {"version": "LabelPilot-1", "flags": {}, "shapes": [],
                        "imagePath": item["relative"], "imageData": None,
                        "imageWidth": width, "imageHeight": height, "checked": False, "description": ""}
            version, status = 0, "unstarted"
        if (document.get("imageWidth"), document.get("imageHeight")) != (width, height):
            raise Problem("原生标注尺寸与原图不一致", 409)
        return {"item": {key: value for key, value in item.items() if key != "path"},
                "document": document, "version": version, "status": status,
                "width": width, "height": height, "image_sha256": image_digest,
                "source_fingerprint": source_digest}

    def validate(self, project, document, width, height, status, confirmed_empty):
        if not isinstance(document, dict):
            raise Problem("标注文档必须是对象", 422)
        shapes = document.get("shapes")
        if not isinstance(shapes, list) or len(shapes) > 10000:
            raise Problem("shapes 必须是数组且不超过 10000 个对象", 422)
        if (document.get("imageWidth"), document.get("imageHeight")) != (width, height):
            raise Problem("不能修改图片尺寸", 422)
        for shape in shapes:
            if not isinstance(shape, dict) or shape.get("label") not in project.names:
                raise Problem("存在项目字典之外的类别；不会自动改类", 422)
            kind = shape.get("shape_type")
            points = shape.get("points")
            if kind not in ("rectangle", "polygon") or not isinstance(points, list):
                raise Problem("仅支持矩形和多边形标注", 422)
            if (kind == "rectangle" and len(points) not in (2, 4)) or (kind == "polygon" and len(points) < 3):
                raise Problem("标注点数量无效", 422)
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise Problem("坐标必须是 [x, y]", 422)
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point):
                    raise Problem("坐标必须是有限数值", 422)
                if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
                    raise Problem("标注超出原图边界；不会静默裁切", 422)
            if max(point[0] for point in points) <= min(point[0] for point in points) or max(point[1] for point in points) <= min(point[1] for point in points):
                raise Problem("标注面积必须大于零", 422)
            if kind == "polygon":
                signed_area = sum(points[index][0] * points[(index + 1) % len(points)][1] -
                                  points[(index + 1) % len(points)][0] * points[index][1] for index in range(len(points)))
                if abs(signed_area) < 1e-8:
                    raise Problem("多边形面积必须大于零", 422)
        if status == "reviewed" and not shapes and not confirmed_empty:
            raise Problem("空标注不能自动作为负样本；请明确确认此图没有目标", 422)

    def save(self, project_id, item_id, payload, predicted=False):
        project = self.project(project_id)
        if payload.get("status") not in (("predicted",) if predicted else ("draft", "reviewed", "skipped")):
            raise Problem("无效的审核状态", 422)
        with self.lock:
            current = self.detail(project_id, item_id)
            if payload.get("expected_version") != current["version"]:
                raise Problem("标注已在其他窗口更新，请重新载入后再编辑", 409)
            if payload.get("image_sha256") != current["image_sha256"]:
                raise Problem("原图指纹不一致", 409)
            if current["version"] == 0 and payload.get("source_fingerprint") != current["source_fingerprint"]:
                raise Problem("原生标注已被其他工具更新，请重新载入", 409)
            if predicted and current["status"] != "unstarted":
                raise Problem("批量预标注不能覆盖任何已有标注", 409)
            document = copy.deepcopy(payload.get("document", {}))
            status = payload["status"]
            self.validate(project, document, current["width"], current["height"], status, payload.get("confirmed_empty", False))
            version = current["version"] + 1
            timestamp = now()
            previous_meta = current["document"].get("labelpilot", {})
            provenance = copy.deepcopy(document.get("labelpilot", {}).get("prediction"))
            document["labelpilot"] = {**previous_meta, "status": status, "revision": version,
                                      "updated_at": timestamp, "image_sha256": current["image_sha256"],
                                      "source_annotation": current["item"].get("source"),
                                      "source_fingerprint": current["source_fingerprint"],
                                      "negative_sample_confirmed": status == "reviewed" and not document["shapes"],
                                      "human_reviewed": status == "reviewed",
                                      "prediction": provenance}
            document["checked"] = status == "reviewed"
            document["imageData"] = None
            if isinstance(document.get("preannotation"), dict):
                document["preannotation"]["human_reviewed"] = status == "reviewed"
                document["preannotation"]["negative_sample_confirmed"] = status == "reviewed" and not document["shapes"]
                document["preannotation"]["status"] = "human_reviewed" if status == "reviewed" else "model_prediction_unreviewed"
            encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT version FROM annotations WHERE project=? AND item=?", (project_id, item_id)).fetchone()
                if (row["version"] if row else 0) != current["version"]:
                    raise Problem("标注已被另一个进程更新，请重新载入", 409)
                if not row:
                    baseline = json.dumps(current["document"], ensure_ascii=False, allow_nan=False)
                    connection.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?)",
                                       (project_id, item_id, 0, current["status"], baseline, timestamp))
                values = (project_id, item_id, version, status, encoded, timestamp)
                connection.execute("INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?, ?)", values)
                connection.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?)", values)
            return {"version": version, "status": status, "document": document, "updated": timestamp}

    def history(self, project_id, item_id):
        self.project(project_id).item(item_id)
        with self.connect() as connection:
            rows = connection.execute("SELECT version, status, updated FROM revisions WHERE project=? AND item=? ORDER BY version DESC",
                                      (project_id, item_id)).fetchall()
        return [dict(row) for row in rows]

    def restore(self, project_id, item_id, revision, expected_version):
        with self.connect() as connection:
            row = connection.execute("SELECT document FROM revisions WHERE project=? AND item=? AND version=?",
                                     (project_id, item_id, revision)).fetchone()
        if not row:
            raise Problem("历史版本不存在", 404)
        current = self.detail(project_id, item_id)
        return self.save(project_id, item_id, {**current, "expected_version": expected_version,
                                               "document": json.loads(row["document"]), "status": "draft"})
