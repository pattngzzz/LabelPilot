import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .ocr import validate_ocr_shape
from .tasks import BACKGROUND, HAND_KEYPOINTS, HAND_SKELETON, MAX_MASK_PIXELS, SHAPE_TYPES, is_rotated_box, polygon_area, read_mask, rectangle_corners, semantic_mask, validate_hand


STATUSES = ("unstarted", "imported", "predicted", "draft", "reviewed", "skipped", "error")
PENDING_REVIEW = ("imported", "predicted", "draft")
AUDITABLE = ("imported", "draft", "reviewed")
REVIEW_FILTERS = {"review": None, "review_empty": "empty", "review_difficult": "difficult", "review_low": "low_confidence"}
REVIEW_CONFIDENCE = 0.5
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


def read_annotation_snapshot(path):
    contents = Path(path).read_bytes()
    return json.loads(contents.decode("utf-8-sig")), hashlib.sha256(contents).hexdigest()


def review_info(document):
    shapes = document["shapes"]
    objects = [shape for shape in shapes if isinstance(shape, dict)]
    scores = [shape.get("score") for shape in objects]
    confidence = min((score for score in scores if type(score) in (int, float) and 0 <= score < REVIEW_CONFIDENCE), default=None)
    metadata = document.get("labelpilot")
    empty = not shapes or (isinstance(metadata, dict) and metadata.get("semantic_empty") is True)
    return {"empty": empty, "difficult": any(shape.get("difficult") is True for shape in objects),
            "low_confidence": confidence is not None, "confidence": confidence}


def review_priority(item):
    info = item["review"]
    return (not info["empty"], not info["difficult"], info["confidence"] if info["low_confidence"] else 1)


def source_group(value):
    if isinstance(value, str):
        name = value.strip()
        if name and name.casefold() != "unassigned":
            return name
    return None


def digest(path, *, checkpoint=None):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if checkpoint:
                checkpoint()
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                # Windows readers briefly hold a sharing lock during progress polling.
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(.02 * (attempt + 1))
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
    def __init__(self, config, intake=None):
        self.config = copy.deepcopy(config)
        self.id = config["id"]
        self.root = Path(config["image_root"]).resolve()
        self.task_type = config.get("task_type", "detect")
        if not isinstance(self.task_type, str) or self.task_type not in SHAPE_TYPES:
            raise Problem("请选择有效的项目任务类型", 422)
        self.names = load_classes(config)
        self.label_aliases = copy.deepcopy(config.get("label_aliases", {}))
        if not isinstance(self.label_aliases, dict) or any(
                not isinstance(alias, str) or not alias.strip() or alias in self.names or
                not isinstance(target, str) or target not in self.names
                for alias, target in self.label_aliases.items()):
            raise Problem("标签别名必须从旧名称直接映射到现有类别，且不能覆盖现有类别", 422)
        self.model_label_aliases = copy.deepcopy(config.get("model_label_aliases", {}))
        if not isinstance(self.model_label_aliases, dict) or any(
                not isinstance(alias, str) or not alias.strip() or
                not isinstance(target, str) or target not in self.names or alias == target or
                (alias in self.label_aliases and self.label_aliases[alias] != target)
                for alias, target in self.model_label_aliases.items()):
            raise Problem("模型标签别名必须直接映射到现有类别，且不能与标注别名冲突", 422)
        if self.task_type == "semantic" and (BACKGROUND in self.names or len(self.names) > 254):
            raise Problem("语义分割最多支持 254 个前景类别，background 为保留类别", 422)
        self.intake = intake or {}
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
        selected = self.config.get("image_files")
        if selected is not None:
            if manifest_path:
                raise Problem("指定图片文件不能同时使用任务清单，请选择其中一种导入方式", 422)
            if not isinstance(selected, list) or not 1 <= len(selected) <= 10000:
                raise Problem("请选择 1 到 10000 个图片文件", 422)
            for value in selected:
                if not isinstance(value, str) or not value.strip() or "\x00" in value:
                    raise Problem("图片文件路径必须为非空文本", 422)
                path = Path(value).expanduser()
                try:
                    path = (path if path.is_absolute() else self.root / path).resolve(strict=True)
                    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                        raise Problem("请选择支持的图片文件，而不是目录或其他格式", 422)
                except (OSError, RuntimeError, ValueError) as error:
                    raise Problem("选中的图片文件不存在或无权访问，请重新选择", 422) from error
                if not path.is_relative_to(self.root):
                    raise Problem("选中的图片必须位于指定图片目录内", 422)
                self.add_item(path, {})
        elif manifest_path:
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
        registered = {item["relative"] for item in self.items}
        for relative in self.intake.keys() - registered:
            path = (self.root / relative).resolve()
            if path.is_file() and path.is_relative_to(self.root) and path.relative_to(self.root).as_posix() == relative:
                self.add_item(path, {})
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
                    if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
                        raise ValueError("缺少 shapes 数组")
                    is_prediction = bool(document.get("preannotation")) or photo_id not in native
                    item.update(source=str(source), status="predicted" if is_prediction else "imported",
                                count=len(document["shapes"]), source_kind="model" if is_prediction else "human",
                                review=review_info(document))
                except (ValueError, OSError, TypeError) as exc:
                    item.update(status="error", error=f"标注读取失败：{exc}")
        self.items.sort(key=lambda item: (item["batch"], item["photo_id"], item["relative"]))
        if not self.items:
            self.warnings.append("未找到图片；请确认图片目录或任务清单路径")
        self.by_id = {item["id"]: item for item in self.items}

    def normalize_labels(self, document):
        document = copy.deepcopy(document)
        if isinstance(document, dict) and isinstance(document.get("shapes"), list):
            for shape in document["shapes"]:
                if isinstance(shape, dict) and isinstance(shape.get("label"), str):
                    shape["label"] = self.label_aliases.get(shape["label"], shape["label"])
        return document

    def add_item(self, path, task):
        if not path.is_relative_to(self.root):
            raise Problem("图片链接指向项目目录之外")
        relative = path.relative_to(self.root).as_posix()
        receipt = self.intake.get(relative)
        if receipt:
            task = {**task, "batch": receipt["batch"], "sha256": receipt["sha256"],
                    "width": receipt["width"], "height": receipt["height"]}
        item_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
        if item_id in self.by_id:
            raise Problem(f"图片清单重复：{relative}")
        self.by_id[item_id] = True
        frame_index = task.get("frame_index")
        self.items.append({
            "id": item_id, "photo_id": task.get("photo_id", path.stem), "path": str(path),
            "relative": relative, "width": task.get("width"), "height": task.get("height"),
            "image_sha256": task.get("sha256"), "batch": task.get("batch") or path.parent.name,
            "source_group": source_group(task.get("source_group")) or "unassigned",
            "source_id": task.get("source_id", ""), "note": task.get("visual_note", ""),
            "frame_index": frame_index if type(frame_index) is int and frame_index >= 0 else None,
            "status": "unstarted", "count": 0, "source": None, "source_kind": None,
        })

    def item(self, item_id):
        try:
            return self.by_id[item_id]
        except KeyError:
            raise Problem("图片不存在", 404)

    def summary(self):
        result = {"id": self.id, "name": self.config["name"], "image_root": str(self.root), "task_type": self.task_type,
                "classes": [{"id": index, "name": name, "display": name,
                             "color": COLORS[index % len(COLORS)]} for index, name in enumerate(self.names)],
                "model_path": self.config.get("model_path", ""), "total": len(self.items),
                "batches": sorted({item["batch"] for item in self.items}),
                "has_source_groups": any(item["source_group"] != "unassigned" for item in self.items),
                "reuse_annotations": self.config.get("reuse_annotations", True),
                "warnings": self.warnings}
        if self.task_type == "hand_pose":
            result["keypoints"] = [{"name": name, "label": label} for name, label in HAND_KEYPOINTS]
            result["skeleton"] = HAND_SKELETON
        return result


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
                CREATE TABLE IF NOT EXISTS similarity_selection (
                    project TEXT NOT NULL, batch TEXT NOT NULL, excluded TEXT NOT NULL,
                    PRIMARY KEY (project, batch)
                );
                CREATE TABLE IF NOT EXISTS contour_suggestions (
                    project TEXT NOT NULL, item TEXT NOT NULL, version INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, image_sha256 TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (project, item)
                );
                CREATE TABLE IF NOT EXISTS annotation_audits (
                    project TEXT NOT NULL, item TEXT NOT NULL, version INTEGER NOT NULL,
                    summary TEXT NOT NULL, evidence TEXT NOT NULL,
                    PRIMARY KEY (project, item)
                );
                CREATE TABLE IF NOT EXISTS intake_receipts (
                    project TEXT NOT NULL, relative TEXT NOT NULL, batch TEXT NOT NULL,
                    sha256 TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, included INTEGER NOT NULL,
                    PRIMARY KEY (project, relative)
                );
                CREATE TABLE IF NOT EXISTS review_routes (
                    project TEXT NOT NULL, item TEXT NOT NULL, version INTEGER NOT NULL,
                    result TEXT NOT NULL, sample INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (project, item)
                );
            """)
        self.configs = read_json(self.registry_path) if self.registry_path.exists() else []
        from .project_management import ProjectManagement
        self.management = ProjectManagement(self)
        lifecycle = self.management.states()
        self.configs = [config for config in self.configs if lifecycle.get(config["id"], {}).get("state") != "deleted"]
        self.project_errors = []
        for config in self.configs:
            if lifecycle.get(config["id"], {}).get("state") == "archived":
                continue
            try:
                self.projects[config["id"]] = self.build_project(config)
            except (Problem, OSError, ValueError, KeyError) as exc:
                self.project_errors.append({"id": config.get("id"), "name": config.get("name"), "error": str(getattr(exc, "message", exc))})

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

    def intake_receipts(self, project_id):
        with self.connect() as connection:
            return {row["relative"]: dict(row) for row in connection.execute(
                "SELECT relative, batch, sha256, width, height, included FROM intake_receipts WHERE project=?", (project_id,)
            )}

    def build_project(self, config):
        intake = self.intake_receipts(config["id"])
        return Project(config, {relative: row for relative, row in intake.items() if row["included"]})

    def import_batch(self, project_id, batch, images):
        with self.lock:
            project = self.project(project_id)
            receipts = self.intake_receipts(project_id)
            overlay = self.overlay(project_id, include_documents=True)
            known = {row["sha256"] for row in receipts.values()}
            known.update(item["image_sha256"] for item in project.items
                         if item["status"] != "unstarted" and item["image_sha256"])
            known.update(row["document"].get("labelpilot", {}).get("image_sha256") for row in overlay.values())
            existing = {item["relative"] for item in project.items
                        if overlay.get(item["id"], item)["status"] != "unstarted"}
            new = {}
            for image in images:
                relative, sha256 = image["relative"], image["sha256"]
                if relative in receipts or relative in existing:
                    continue
                new[relative] = {**image, "batch": batch, "included": sha256 not in known}
                known.add(sha256)
            if new:
                combined = {**receipts, **new}
                replacement = Project(project.config, {relative: row for relative, row in combined.items() if row["included"]})
                with self.connect() as connection:
                    connection.executemany("INSERT OR IGNORE INTO intake_receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
                                           ((project_id, relative, batch, row["sha256"], row["width"], row["height"], row["included"])
                                            for relative, row in new.items()))
                self.projects[project_id] = replacement
                receipts = combined
        imported = [row for row in receipts.values() if row["batch"] == batch]
        return {"included": sum(row["included"] for row in imported), "duplicates": sum(not row["included"] for row in imported)}

    def register(self, config):
        config = copy.deepcopy(config)
        if type(config.get("reuse_annotations", True)) is not bool:
            raise Problem("标注复用开关必须为布尔值", 422)
        config["id"] = uuid.uuid4().hex[:12]
        if config.get("task_type") == "hand_pose" and not (config.get("classes") or config.get("class_map_path")):
            config["classes"] = ["hand"]
        if config.get("task_type") == "ocr" and not (config.get("classes") or config.get("class_map_path")):
            config["classes"] = ["text"]
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
        if config.get("image_files") is not None:
            config["image_files"] = [item["relative"] for item in project.items]
            project.config["image_files"] = list(config["image_files"])
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
            replacement = self.build_project(project.config)
            self.projects[project_id] = replacement
        return replacement.summary()

    def add_label(self, project_id, name):
        return self._add_labels(project_id, [name])["project"]

    def import_labels(self, project_id, filename, content):
        if not isinstance(filename, str) or Path(filename).suffix.lower() not in (".txt", ".json"):
            raise Problem("请选择 classes.txt 或 class_map.json 标签文件", 422)
        if not isinstance(content, str):
            raise Problem("标签文件必须为 UTF-8 文本，且不能超过 1 MB", 422)
        try:
            size = len(content.encode("utf-8"))
        except UnicodeEncodeError:
            raise Problem("标签文件必须为有效的 UTF-8 文本", 422)
        if size > 1024 * 1024:
            raise Problem("标签文件不能超过 1 MB", 422)
        content = content.lstrip("\ufeff")
        if Path(filename).suffix.lower() == ".txt":
            names = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            try:
                value = json.loads(content)
            except (ValueError, RecursionError):
                raise Problem("标签 JSON 格式无效", 422)
            names = value.get("names", value) if isinstance(value, dict) else value
            if isinstance(names, dict):
                if set(names) != {str(index) for index in range(len(names))}:
                    raise Problem("类别 ID 必须从 0 连续排列", 422)
                names = [names[str(index)] for index in range(len(names))]
        if not isinstance(names, list) or not names:
            raise Problem("标签文件为空或格式无效；请使用每行一个标签的 TXT 或 names 格式的 JSON", 422)
        return self._add_labels(project_id, names, allow_aliases=True)

    def _add_labels(self, project_id, names, *, allow_aliases=False):
        normalized = []
        for name in names:
            if not isinstance(name, str):
                raise Problem("标签名称必须是文本", 422)
            name = name.strip()
            if not name or len(name) > 160 or any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in name):
                raise Problem("标签名称必须为 1 到 160 个字符，不能包含换行或控制字符", 422)
            normalized.append(name)
        with self.lock:
            project = self.project(project_id)
            added = []
            existing = set(project.names)
            for name in normalized:
                if name in project.label_aliases:
                    if not allow_aliases:
                        raise Problem(f"该标签已更名为 {project.label_aliases[name]}，请使用新名称", 422)
                    name = project.label_aliases[name]
                if name not in existing:
                    added.append(name)
                    existing.add(name)
            if project.task_type == "semantic" and (BACKGROUND in added or len(existing) > 254):
                raise Problem("语义分割最多支持 254 个前景类别，background 为保留类别", 422)
            if added:
                configs = copy.deepcopy(self.configs)
                config = next(config for config in configs if config["id"] == project_id)
                if config.get("class_map_path"):
                    config["imported_class_map_path"] = config.pop("class_map_path")
                config["classes"] = project.names + added
                atomic_json(self.registry_path, configs)
                self.configs = configs
                project.config = copy.deepcopy(config)
                project.names = config["classes"]
            return {"project": project.summary(), "added": len(added), "skipped": len(normalized) - len(added)}

    def delete_label(self, project_id, name, expected_classes):
        if not isinstance(name, str) or not name:
            raise Problem("请选择要删除的标签", 422)
        with self.lock:
            project = self.project(project_id)
            if expected_classes != project.names:
                raise Problem("类别表已变化，请刷新页面后重试", 409)
            if name not in project.names:
                raise Problem("标签不存在", 404)
            if len(project.names) <= 1:
                raise Problem("项目至少需要保留一个标签，不能删除最后一个标签", 422)

            def check_document(document, location):
                if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
                    raise Problem(f"{location}无法校验，已阻止删除标签", 409)
                for shape in document["shapes"]:
                    if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
                        raise Problem(f"{location}含无效标注，已阻止删除标签", 409)
                    if project.label_aliases.get(shape["label"], shape["label"]) == name:
                        raise Problem(f"标签 {name} 仍被{location}引用，不能删除；只允许删除未使用标签，不会连带删除标注框", 409)

            with self.connect() as connection:
                for table, location in (("annotations", "已保存标注"), ("revisions", "历史标注")):
                    for row in connection.execute(f"SELECT document FROM {table} WHERE project=?", (project_id,)):
                        try:
                            document = json.loads(row["document"])
                        except (ValueError, TypeError):
                            raise Problem(f"{location}无法读取，已阻止删除标签", 409)
                        check_document(document, location)
            try:
                sources = set()
                for field in ("annotation_root", "preannotation_root"):
                    sources.update(project.source_map(project.config.get(field)).values())
                for source in sources:
                    check_document(read_json(source), "源标注")
            except (OSError, ValueError, TypeError):
                raise Problem("源标注无法读取，已阻止删除标签；请先检查源文件", 409)
            configs = copy.deepcopy(self.configs)
            config = next(config for config in configs if config["id"] == project_id)
            if config.get("class_map_path"):
                config["imported_class_map_path"] = config.pop("class_map_path")
            config["classes"] = [label for label in project.names if label != name]
            aliases = {alias: target for alias, target in project.label_aliases.items() if target != name}
            if "label_aliases" in config:
                config["label_aliases"] = aliases
            model_aliases = {alias: target for alias, target in project.model_label_aliases.items() if target != name}
            if "model_label_aliases" in config:
                config["model_label_aliases"] = model_aliases
            atomic_json(self.registry_path, configs)
            self.configs = configs
            project.config = copy.deepcopy(config)
            project.names = list(config["classes"])
            project.label_aliases = aliases
            project.model_label_aliases = model_aliases
            return project.summary()

    def overlay(self, project_id, include_documents=False, include_review=False):
        overlays = {}
        with self.connect() as connection:
            rows = connection.execute("""
                SELECT item, version, status, updated,
                       CASE WHEN json_type(document, '$.shapes') = 'array'
                            THEN json_array_length(document, '$.shapes') END AS object_count,
                       CASE WHEN ? OR (? AND status IN (?, ?, ?)) THEN document END AS document
                FROM annotations WHERE project=?
            """, (include_documents, include_review, *PENDING_REVIEW, project_id))
            for row in rows:
                if row["object_count"] is None:
                    raise Problem("已保存标注的 shapes 不是有效数组，已阻止读取队列", 409)
                item = {"version": row["version"], "status": row["status"],
                        "count": row["object_count"], "updated": row["updated"]}
                document = json.loads(row["document"]) if row["document"] is not None else None
                if include_documents:
                    item["document"] = self.project(project_id).normalize_labels(document)
                if include_review and row["status"] in PENDING_REVIEW:
                    item["review"] = review_info(document)
                overlays[row["item"]] = item
        return overlays

    def listing(self, project_id, query="", batch="", status="", offset=0, limit=80, *, include_documents=False):
        project = self.project(project_id)
        review_queue = status in REVIEW_FILTERS
        reason = REVIEW_FILTERS.get(status)
        overlays = self.overlay(project_id, include_documents=include_documents, include_review=review_queue)
        excluded = self.similarity_exclusions(project_id, batch) if status == "curated" else set()
        audits = self.audit_summaries(project_id) if status == "audit" else {}
        routed_queue = status in ("triage", "spot_review", "routable")
        routes = self.review_routes(project_id) if routed_queue else {}
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
            if review_queue:
                if item["status"] not in PENDING_REVIEW or (reason and not item["review"][reason]):
                    continue
            else:
                item.pop("review", None)
                if status == "curated":
                    if item["status"] != "unstarted" or item["id"] in excluded:
                        continue
                elif status == "audit":
                    finding = audits.get(item["id"])
                    if not finding or item["status"] not in AUDITABLE or finding["version"] != item.get("version", 0):
                        continue
                    item["audit"] = finding
                elif status == "auditable":
                    if item["status"] not in AUDITABLE:
                        continue
                elif routed_queue:
                    route = routes.get(item["id"])
                    if item["status"] not in PENDING_REVIEW:
                        continue
                    if status == "routable":
                        if route and route["version"] == item.get("version", 0):
                            continue
                    elif not route or (status == "spot_review" and not route["sample"]):
                        continue
                    if route:
                        item["route"] = route
                elif status and item["status"] != status:
                    continue
            filtered.append(item)
        if review_queue:
            filtered.sort(key=review_priority)
        if status == "audit":
            filtered.sort(key=lambda item: (-item["audit"]["confidence"], -item["audit"]["count"]))
        if status == "triage":
            filtered.sort(key=lambda item: -item["route"]["priority"])
        return {"items": filtered[offset:offset + limit], "filtered": len(filtered),
                "total": len(project.items), "counts": counts, "offset": offset, "limit": limit}

    def review_routes(self, project_id):
        with self.connect() as connection:
            return {row["item"]: {"version": row["version"], "sample": bool(row["sample"]), **json.loads(row["result"])}
                    for row in connection.execute("SELECT item, version, result, sample FROM review_routes WHERE project=?", (project_id,))}

    def save_review_route(self, project_id, item_id, version, result):
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO review_routes VALUES (?, ?, ?, ?, 0)",
                               (project_id, item_id, version, json.dumps(result, ensure_ascii=False)))

    def set_review_sample(self, project_id, ids, samples):
        with self.connect() as connection:
            connection.executemany("UPDATE review_routes SET sample=? WHERE project=? AND item=?",
                                   ((item_id in samples, project_id, item_id) for item_id in ids))

    def similarity_exclusions(self, project_id, batch):
        with self.connect() as connection:
            row = connection.execute("SELECT excluded FROM similarity_selection WHERE project=? AND batch=?",
                                     (project_id, batch)).fetchone()
        return set(json.loads(row["excluded"])) if row else set()

    def save_similarity_selection(self, project_id, batch, ids):
        project = self.project(project_id)
        if not isinstance(batch, str) or not isinstance(ids, list) or len(ids) > 10000:
            raise Problem("相似图筛选范围无效", 422)
        if any(not isinstance(item_id, str) or item_id not in project.by_id or
               (batch and project.by_id[item_id]["batch"] != batch) for item_id in ids):
            raise Problem("相似图不在所选批次内，请重新扫描", 422)
        excluded = sorted(set(ids))
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO similarity_selection VALUES (?, ?, ?)",
                               (project_id, batch, json.dumps(excluded)))
        return {"excluded": len(excluded)}

    def audit_summaries(self, project_id):
        with self.connect() as connection:
            return {row["item"]: {"version": row["version"], **json.loads(row["summary"])}
                    for row in connection.execute("SELECT item, version, summary FROM annotation_audits WHERE project=?", (project_id,))}

    def audit_detail(self, project_id, item_id):
        self.project(project_id).item(item_id)
        with self.connect() as connection:
            row = connection.execute("SELECT version, evidence FROM annotation_audits WHERE project=? AND item=?",
                                     (project_id, item_id)).fetchone()
        if not row:
            return None
        current = self.detail(project_id, item_id)
        evidence = json.loads(row["evidence"])
        if (current["version"], current["image_sha256"], current["source_fingerprint"]) != (
            row["version"], evidence["image_sha256"], evidence["source_fingerprint"]
        ):
            return None
        return evidence

    def save_audit(self, project_id, item_id, evidence, summary):
        with self.connect() as connection:
            if summary:
                connection.execute("INSERT OR REPLACE INTO annotation_audits VALUES (?, ?, ?, ?, ?)",
                                   (project_id, item_id, evidence["version"], json.dumps(summary), json.dumps(evidence, ensure_ascii=False)))
            else:
                connection.execute("DELETE FROM annotation_audits WHERE project=? AND item=?", (project_id, item_id))

    def dismiss_audit(self, project_id, item_id, evidence_id):
        self.project(project_id).item(item_id)
        if not isinstance(evidence_id, str):
            raise Problem("请重新载入疑点结果", 422)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT evidence FROM annotation_audits WHERE project=? AND item=?",
                                     (project_id, item_id)).fetchone()
            if row and json.loads(row["evidence"]).get("id") != evidence_id:
                raise Problem("疑点已重新扫描，请刷新后复核", 409)
            connection.execute("DELETE FROM annotation_audits WHERE project=? AND item=?", (project_id, item_id))
        return {"dismissed": True}

    def image_info(self, project, item, *, verify_image=False):
        path = Path(item["path"])
        if not path.is_file() or not path.resolve().is_relative_to(project.root):
            raise Problem("原图不存在或路径已改变", 409)
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274, 1) != 1:
                raise Problem("此图含 EXIF 旋转，需先生成方向归一化的独立数据副本，避免标注错位", 422)
            if verify_image:
                image.load()
        if item["width"] and (width, height) != (item["width"], item["height"]):
            raise Problem("原图尺寸与任务清单不符", 409)
        image_digest = digest(path)
        if item["image_sha256"] and image_digest != item["image_sha256"]:
            raise Problem("原图 SHA-256 与任务清单不符；已阻止编辑", 409)
        return width, height, image_digest

    def detail(self, project_id, item_id, *, verify_image=False):
        project = self.project(project_id)
        item = project.item(item_id)
        if item.get("error"):
            raise Problem(item["error"], 422)
        width, height, image_digest = self.image_info(project, item, verify_image=verify_image)
        source_digest = None
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM annotations WHERE project=? AND item=?", (project_id, item_id)).fetchone()
        if row:
            document = json.loads(row["document"])
            source_digest = digest(item["source"]) if item["source"] else None
            version, status = row["version"], row["status"]
            recorded = document.get("labelpilot", {}).get("image_sha256")
            if recorded and recorded != image_digest:
                raise Problem("原图已在上次保存后改变；已阻止旧标注套用", 409)
        elif item["source"]:
            document, source_digest = read_annotation_snapshot(item["source"])
            version, status = 0, item["status"]
        else:
            document = {"version": "LabelPilot-1", "flags": {}, "shapes": [],
                        "imagePath": item["relative"], "imageData": None,
                        "imageWidth": width, "imageHeight": height, "checked": False, "description": ""}
            version, status = 0, "unstarted"
        if (document.get("imageWidth"), document.get("imageHeight")) != (width, height):
            raise Problem("原生标注尺寸与原图不一致", 409)
        document = project.normalize_labels(document)
        if not row and item["source"] and project.task_type != "detect":
            self.validate(project, document, width, height, "draft", False)
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
        task = project.task_type
        if task == "semantic" and width * height > MAX_MASK_PIXELS:
            raise Problem("语义掩膜最多支持 3200 万像素", 422)
        for shape in shapes:
            if not isinstance(shape, dict) or shape.get("label") not in project.names:
                raise Problem("存在项目字典之外的类别；不会自动改类", 422)
            kind = shape.get("shape_type")
            if task == "segment" and kind == "rectangle" and status == "reviewed":
                raise Problem("实例分割审核不支持此标注形状：请先将矩形框补成轮廓，或保存为草稿", 422)
            points = shape.get("points")
            if kind not in SHAPE_TYPES[task] or not isinstance(points, list):
                raise Problem("当前项目任务不支持此标注形状", 422)
            if ((kind == "rectangle" and len(points) not in (2, 4))
                    or (kind in ("mask", "hand") and len(points) != 2)
                    or (kind == "polygon" and len(points) < 3)
                    or (kind == "brush" and not 1 <= len(points) <= 4096)):
                raise Problem("标注点数量无效", 422)
            allow_outside = task in ("detect", "segment", "ocr") and is_rotated_box(shape)
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise Problem("坐标必须是 [x, y]", 422)
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point):
                    raise Problem("坐标必须是有限数值", 422)
                if not allow_outside and (not 0 <= point[0] <= width or not 0 <= point[1] <= height):
                    raise Problem("标注超出原图边界；不会静默裁切", 422)
            if kind == "brush":
                radius = shape.get("radius")
                if type(radius) is not int or not 1 <= radius <= 256 or type(shape.get("erase", False)) is not bool:
                    raise Problem("笔刷半径必须为 1 到 256 像素的整数", 422)
                continue
            if shape.get("erase"):
                raise Problem("橡皮仅支持笔刷图层", 422)
            xs, ys = zip(*points)
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            if right <= left or bottom <= top:
                raise Problem("标注面积必须大于零", 422)
            if kind == "polygon" and polygon_area(points) < 5e-9:
                raise Problem("多边形面积必须大于零", 422)
            if kind == "rectangle" and "rotation" in shape and (not is_rotated_box(shape) or not rectangle_corners(points)):
                raise Problem("旋转矩形需要四个按顺序排列、相邻边垂直的角点", 422)
            try:
                if kind == "mask":
                    read_mask(shape)
                elif kind == "hand":
                    validate_hand(shape, (left, top, right, bottom))
                if task == "ocr":
                    validate_ocr_shape(shape, reviewed=status == "reviewed")
            except ValueError as exc:
                raise Problem(str(exc), 422) from exc
        empty = not shapes
        if task == "semantic" and any(shape.get("erase") for shape in shapes):
            empty = semantic_mask(shapes, project.names, width, height).getbbox() is None
        if status == "reviewed" and empty and confirmed_empty is not True:
            raise Problem("空标注不能自动作为负样本；请明确确认此图没有目标", 422)
        return empty

    def save(self, project_id, item_id, payload, predicted=False):
        return self.save_many(project_id, [(item_id, payload)], predicted=predicted)[0]

    def prepare_save(self, project_id, item_id, payload, predicted):
        project = self.project(project_id)
        if payload.get("status") not in (("predicted",) if predicted else ("draft", "reviewed", "skipped")):
            raise Problem("无效的审核状态", 422)
        current = self.detail(project_id, item_id)
        if payload.get("expected_version") != current["version"]:
            raise Problem("标注已在其他窗口更新，请重新载入后再编辑", 409)
        if payload.get("image_sha256") != current["image_sha256"]:
            raise Problem("原图指纹不一致", 409)
        if current["version"] == 0 and payload.get("source_fingerprint") != current["source_fingerprint"]:
            raise Problem("原生标注已被其他工具更新，请重新载入", 409)
        if predicted and current["status"] != "unstarted":
            raise Problem("批量预标注不能覆盖任何已有标注", 409)
        document = project.normalize_labels(payload.get("document", {}))
        status = payload["status"]
        empty = self.validate(project, document, current["width"], current["height"], status, payload.get("confirmed_empty", False))
        version = current["version"] + 1
        timestamp = now()
        previous_meta = current["document"].get("labelpilot", {})
        provenance = copy.deepcopy(document.get("labelpilot", {}).get("prediction"))
        document["labelpilot"] = {**previous_meta, "status": status, "revision": version,
                                  "updated_at": timestamp, "image_sha256": current["image_sha256"],
                                  "source_annotation": current["item"].get("source"),
                                  "source_fingerprint": current["source_fingerprint"],
                                  "negative_sample_confirmed": status == "reviewed" and empty,
                                  "human_reviewed": status == "reviewed",
                                  "prediction": provenance}
        incoming_meta = payload.get("document", {}).get("labelpilot", {})
        for key in ("sam3_suggestions", "duplicate_cleanup", "annotation_reuse"):
            if key in incoming_meta:
                document["labelpilot"][key] = copy.deepcopy(incoming_meta[key])
        if project.task_type == "semantic":
            document["labelpilot"]["semantic_empty"] = empty
        document["checked"] = status == "reviewed"
        document["imageData"] = None
        if isinstance(document.get("preannotation"), dict):
            document["preannotation"]["human_reviewed"] = status == "reviewed"
            document["preannotation"]["negative_sample_confirmed"] = status == "reviewed" and empty
            document["preannotation"]["status"] = "human_reviewed" if status == "reviewed" else "model_prediction_unreviewed"
        result = {"version": version, "status": status, "document": document, "updated": timestamp}
        return current, result

    def save_many(self, project_id, entries, *, predicted=False):
        with self.lock:
            prepared = [(item_id, *self.prepare_save(project_id, item_id, payload, predicted)) for item_id, payload in entries]
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                for item_id, current, result in prepared:
                    row = connection.execute("SELECT version FROM annotations WHERE project=? AND item=?",
                                             (project_id, item_id)).fetchone()
                    if (row["version"] if row else 0) != current["version"]:
                        raise Problem("标注已被另一个进程更新，请重新载入", 409)
                    if not row:
                        baseline = json.dumps(current["document"], ensure_ascii=False, allow_nan=False)
                        connection.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?)",
                                           (project_id, item_id, 0, current["status"], baseline, result["updated"]))
                    encoded = json.dumps(result["document"], ensure_ascii=False, allow_nan=False)
                    values = (project_id, item_id, result["version"], result["status"], encoded, result["updated"])
                    connection.execute("INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?, ?)", values)
                    connection.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?)", values)
                    self.management.sync_reviewed(connection, self.project(project_id).config, item_id, result)
        return [result for _, _, result in prepared]

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
