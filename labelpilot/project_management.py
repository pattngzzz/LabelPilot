import copy
import hashlib
import json
import re
from contextlib import contextmanager

from .store import Problem, atomic_json, now


class ProjectManagement:
    def __init__(self, store):
        self.store = store
        self.requests = {}
        with store.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS project_lifecycle (
                    project TEXT PRIMARY KEY, state TEXT NOT NULL,
                    summary TEXT NOT NULL, updated TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotation_library (
                    project TEXT NOT NULL, item TEXT NOT NULL, name TEXT NOT NULL,
                    task TEXT NOT NULL, sha256 TEXT NOT NULL, width INTEGER NOT NULL,
                    height INTEGER NOT NULL, version INTEGER NOT NULL,
                    document TEXT NOT NULL, updated TEXT NOT NULL,
                    PRIMARY KEY (project, item)
                );
                CREATE INDEX IF NOT EXISTS annotation_library_image ON annotation_library (sha256, task);
                CREATE TABLE IF NOT EXISTS project_management_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            if not connection.execute("SELECT 1 FROM project_management_meta WHERE key='library_v1'").fetchone():
                configs = {config["id"]: config for config in store.configs}
                for row in connection.execute("SELECT * FROM annotations WHERE status='reviewed'").fetchall():
                    config = configs.get(row["project"])
                    if config:
                        self.sync_reviewed(connection, config, row["item"], dict(row))
                connection.execute("INSERT INTO project_management_meta VALUES ('library_v1', '1')")

    def states(self):
        with self.store.connect() as connection:
            return {row["project"]: dict(row) for row in connection.execute("SELECT * FROM project_lifecycle")}

    @contextmanager
    def access(self, project_id):
        with self.store.lock:
            self.store.project(project_id)
            self.requests[project_id] = self.requests.get(project_id, 0) + 1
        try:
            yield
        finally:
            with self.store.lock:
                self.requests[project_id] -= 1
                if not self.requests[project_id]:
                    del self.requests[project_id]

    def config(self, project_id):
        config = next((row for row in self.store.configs if row["id"] == project_id), None)
        if not config or self.states().get(project_id, {}).get("state") == "deleted":
            raise Problem("项目不存在或已永久删除", 404)
        return config

    def listing(self):
        with self.store.lock:
            states = self.states()
            with self.store.connect() as connection:
                counts = {row["project"]: row["count"] for row in connection.execute(
                    "SELECT project, COUNT(*) AS count FROM annotations WHERE status='reviewed' GROUP BY project")}
            result = []
            for config in self.store.configs:
                lifecycle = states.get(config["id"], {})
                if lifecycle.get("state") == "deleted":
                    continue
                project = self.store.projects.get(config["id"])
                snapshot = json.loads(lifecycle["summary"]) if lifecycle else {}
                result.append({"id": config["id"], "name": config["name"], "image_root": config["image_root"],
                               "task_type": config.get("task_type", "detect"),
                               "state": lifecycle.get("state", "active"),
                               "available": project is not None, "total": len(project.items) if project else snapshot.get("total"),
                               "reviewed": counts.get(config["id"], 0), "updated": lifecycle.get("updated")})
            return {"projects": result}

    def archive(self, project_id):
        with self.store.lock:
            config = self.config(project_id)
            if self.requests.get(project_id):
                raise Problem("项目仍有读写请求，请稍后归档", 409)
            if self.states().get(project_id, {}).get("state") == "archived":
                raise Problem("项目已经归档", 409)
            project = self.store.projects.get(project_id)
            summary = project.summary() if project else {"id": project_id, "name": config["name"], "total": None}
            with self.store.connect() as connection:
                connection.execute("INSERT OR REPLACE INTO project_lifecycle VALUES (?, 'archived', ?, ?)",
                                   (project_id, json.dumps(summary, ensure_ascii=False), now()))
            self.store.projects.pop(project_id, None)
            self.store.project_errors = [row for row in self.store.project_errors if row.get("id") != project_id]
            return {"id": project_id, "state": "archived"}

    def restore(self, project_id):
        with self.store.lock:
            config = self.config(project_id)
            if self.states().get(project_id, {}).get("state") != "archived":
                raise Problem("只能复原已归档项目", 409)
            project = self.store.build_project(config)
            with self.store.connect() as connection:
                connection.execute("DELETE FROM project_lifecycle WHERE project=?", (project_id,))
            self.store.projects[project_id] = project
            return project.summary()

    def delete(self, project_id, confirmation):
        with self.store.lock:
            config = self.config(project_id)
            lifecycle = self.states().get(project_id, {})
            if lifecycle.get("state") != "archived":
                raise Problem("请先归档项目，再执行永久删除", 409)
            if confirmation != config["name"]:
                raise Problem("请输入完整项目名称以确认永久删除", 422)
            if self.requests.get(project_id):
                raise Problem("项目仍有读写请求，请稍后删除", 409)
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                for row in connection.execute("SELECT * FROM annotations WHERE project=?", (project_id,)).fetchall():
                    self.sync_reviewed(connection, config, row["item"], dict(row))
                for table in ("annotations", "revisions", "similarity_selection", "contour_suggestions",
                              "annotation_audits", "intake_receipts", "review_routes"):
                    connection.execute(f"DELETE FROM {table} WHERE project=?", (project_id,))
                connection.execute("UPDATE project_lifecycle SET state='deleted', updated=? WHERE project=?", (now(), project_id))
                retained = connection.execute("SELECT COUNT(*) FROM annotation_library WHERE project=?", (project_id,)).fetchone()[0]
            self.store.configs = [row for row in self.store.configs if row["id"] != project_id]
            result = {"id": project_id, "state": "deleted", "retained_annotations": retained}
            try:
                atomic_json(self.store.registry_path, self.store.configs)
            except OSError:
                result["warning"] = "项目已删除；配置清理失败，数据库删除标记会阻止项目重新出现"
            return result

    def sync_reviewed(self, connection, config, item_id, result):
        connection.execute("DELETE FROM annotation_library WHERE project=? AND item=?", (config["id"], item_id))
        if result["status"] != "reviewed":
            return
        try:
            document = json.loads(result["document"]) if isinstance(result["document"], str) else result["document"]
            metadata = document.get("labelpilot", {})
            image_hash = metadata.get("image_sha256", "")
            if (metadata.get("human_reviewed") is not True or not re.fullmatch(r"[a-f0-9]{64}", image_hash)
                    or (not document.get("shapes") and metadata.get("negative_sample_confirmed") is not True)):
                return
            width, height = document["imageWidth"], document["imageHeight"]
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                return
            snapshot = self.normalize_source(document, config)
            snapshot["imageData"] = None
            connection.execute("INSERT INTO annotation_library VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (config["id"], item_id, config["name"], config.get("task_type", "detect"), image_hash,
                                width, height, result["version"], json.dumps(snapshot, ensure_ascii=False), result["updated"]))
        except (ValueError, TypeError, KeyError):
            return

    def normalize_source(self, document, config):
        document = copy.deepcopy(document)
        aliases = config.get("label_aliases", {})
        for shape in document.get("shapes", []):
            if isinstance(shape, dict) and isinstance(shape.get("label"), str):
                shape["label"] = aliases.get(shape["label"], shape["label"])
        return document

    def candidates(self, project_id, current, *, documents=False):
        project = self.store.project(project_id)
        configs = {config["id"]: config for config in self.store.configs}
        result = {"candidates": [], "incompatible": []}
        if (current["status"] != "unstarted" or current["version"] != 0 or current["item"].get("source")
                or current["document"].get("shapes")):
            return result
        with self.store.connect() as connection:
            rows = connection.execute("""
                SELECT library.*, annotations.document AS current_document, annotations.status AS current_status,
                       annotations.version AS current_version, annotations.updated AS current_updated,
                       project_lifecycle.state AS source_state
                FROM annotation_library AS library
                LEFT JOIN annotations ON annotations.project=library.project AND annotations.item=library.item
                LEFT JOIN project_lifecycle ON project_lifecycle.project=library.project
                WHERE library.sha256=? AND library.task=? AND library.project!=?
                ORDER BY library.updated DESC, library.project, library.item
            """, (current["image_sha256"], project.task_type, project_id)).fetchall()
        for row in rows:
            if row["current_status"] is not None and row["current_status"] != "reviewed":
                continue
            document = self.normalize_source(json.loads(row["current_document"] or row["document"]), configs.get(row["project"], {}))
            metadata = document.get("labelpilot", {})
            if (metadata.get("human_reviewed") is not True or metadata.get("image_sha256") != current["image_sha256"]
                    or (document.get("imageWidth"), document.get("imageHeight")) != (current["width"], current["height"])):
                continue
            version = row["current_version"] if row["current_version"] is not None else row["version"]
            encoded = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            reference = hashlib.sha256(f"{row['project']}:{row['item']}:{version}:{encoded}".encode()).hexdigest()
            candidate = {"reference": reference, "project": row["project"], "project_name": row["name"],
                         "item": row["item"], "relative": document.get("imagePath", ""), "version": version,
                         "objects": len(document["shapes"]), "empty": metadata.get("negative_sample_confirmed") is True,
                         "updated": row["current_updated"] or row["updated"], "source_deleted": row["source_state"] == "deleted"}
            try:
                self.store.validate(project, copy.deepcopy(document), current["width"], current["height"], "reviewed", candidate["empty"])
            except Problem as exc:
                missing = sorted({shape.get("label", "") for shape in document["shapes"]} - set(project.names))
                result["incompatible"].append({**candidate, "reason": "缺少标签：" + ", ".join(missing) if missing else exc.message})
                continue
            if documents:
                candidate["document"] = document
            result["candidates"].append(candidate)
        return result

    def reuse(self, project_id, item_id, payload):
        with self.store.lock:
            current = self.store.detail(project_id, item_id)
            if payload.get("expected_version") != current["version"] or payload.get("image_sha256") != current["image_sha256"]:
                raise Problem("图片或标注版本已改变，请重新载入", 409)
            candidate = next((row for row in self.candidates(project_id, current, documents=True)["candidates"]
                              if row["reference"] == payload.get("reference")), None)
            if not candidate:
                raise Problem("已有标注不可复用或来源已更新，请重新载入；不会覆盖当前标注", 409)
            document = copy.deepcopy(candidate.pop("document"))
            document["imagePath"] = current["item"]["relative"]
            document["labelpilot"] = {"annotation_reuse": {**candidate, "image_sha256": current["image_sha256"], "reused_at": now()}}
            return self.store.save(project_id, item_id, {"document": document, "expected_version": current["version"],
                                   "image_sha256": current["image_sha256"], "source_fingerprint": current["source_fingerprint"],
                                   "status": "draft"})
