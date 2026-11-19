import copy
import hashlib
import json
import shutil
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from .blindsets import ID, get_blindset, reserve_blind_images
from .curation import SIMILARITY_LEVELS, similar_frames
from .evaluation import weight_path
from .intake import observe_images, source_folder, stratify_review, verify_incoming
from .store import Problem, atomic_json, digest, now, read_json
from .training import dataset_identity, read_state, training_options, training_selection


IN_PROGRESS = ("queued", "running", "waiting")
CHILD_ACTIVE = (*IN_PROGRESS, "preparing", "cancelling")
INTAKE_DEFAULTS = {"folder": "", "watch": False, "similarity": "balanced", "representatives": False,
                   "confidence": .25, "low": .50, "rare": 10, "sample": 10, "companion": ""}
RELEASE_DEFAULTS = {"enabled": False, "blindset": "", "min_images": 50, "min_each": 3,
                    "baseline": "", "eval_confidence": .25, "eval_iou": .50}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def number(payload, key, minimum, maximum, *, integer=False):
    value = payload[key]
    types = (int,) if integer else (float, int)
    if type(value) not in types or not minimum <= value <= maximum:
        raise Problem(f"{key} 必须在 {minimum} 到 {maximum} 之间", 422)
    return value


def recipe_defaults(project):
    return {"intake": dict(INTAKE_DEFAULTS), "release": {**RELEASE_DEFAULTS, **training_options(project, {}),
            "gates": {label: {"recall": .80, "false_positives": 5} for label in project.names}}}


def normalize_recipe(store, project, payload):
    defaults = recipe_defaults(project)
    incoming, release = payload.get("intake", {}), payload.get("release", {})
    if not isinstance(incoming, dict) or not isinstance(release, dict):
        raise Problem("无效的工作流设置", 422)
    intake = {key: incoming.get(key, value) for key, value in defaults["intake"].items()}
    value = {key: release.get(key, default) for key, default in defaults["release"].items()}
    for settings, key in ((intake, "watch"), (intake, "representatives"), (value, "enabled")):
        if type(settings[key]) is not bool:
            raise Problem("自动化开关必须为布尔值", 422)
    folder = source_folder(project, intake["folder"])
    intake["folder"] = folder.relative_to(project.root).as_posix()
    if not isinstance(intake["similarity"], str) or intake["similarity"] not in SIMILARITY_LEVELS:
        raise Problem("请选择有效的相似程度", 422)
    for settings, key, minimum, maximum, integer in (
        (intake, "confidence", .01, 1, False), (intake, "low", .01, 1, False),
        (intake, "rare", 0, 10000, True), (intake, "sample", 0, 100, True),
        (value, "min_images", 2, 10000, True), (value, "min_each", 1, 10000, True),
        (value, "eval_confidence", .01, 1, False), (value, "eval_iou", .01, 1, False),
    ):
        number(settings, key, minimum, maximum, integer=integer)
    for settings, key in ((intake, "companion"), (value, "baseline")):
        if not isinstance(settings[key], str):
            raise Problem("权重路径必须为文本", 422)
        settings[key] = str(weight_path(settings[key])) if settings[key].strip() else ""
    value.update(training_options(project, value))
    blindset = value["blindset"]
    if blindset or value["enabled"]:
        get_blindset(store, project.id, blindset)
    gates = value["gates"]
    if not isinstance(gates, dict) or not gates or set(gates) - set(project.names):
        raise Problem("至少为一个项目类别设置验收门槛", 422)
    for gate in gates.values():
        if not isinstance(gate, dict) or "recall" not in gate or "false_positives" not in gate:
            raise Problem("每个验收类别需提供召回率和误检上限", 422)
        number(gate, "recall", 0, 1)
        number(gate, "false_positives", 0, 1000000, integer=True)
    return {"intake": intake, "release": value}


def assess_gates(report, gates):
    rows = {row["name"]: row for row in report["classes"]}
    results = []
    for name, rule in gates.items():
        current = rows.get(name, {}).get("new", {})
        recall, errors = current.get("recall"), current.get("false_positives")
        cleared = recall is not None and errors is not None and recall >= rule["recall"] and errors <= rule["false_positives"]
        results.append({"name": name, "passed": cleared, "recall": recall, "false_positives": errors, "rule": rule})
    return results


def readiness(rows, settings):
    coverage = Counter()
    for row in rows:
        coverage.update({shape["label"] for shape in row["document"]["shapes"] if shape["shape_type"] == "rectangle"})
    missing = {name: coverage[name] for name in settings["gates"] if coverage[name] < settings["min_each"]}
    return {"images": len(rows), "required": settings["min_images"], "missing": missing,
            "ready": len(rows) >= settings["min_images"] and not missing}


class Automation:
    def __init__(self, store, jobs, training, comparisons):
        self.store, self.jobs, self.training, self.comparisons = store, jobs, training, comparisons
        self.directory = store.data / "automation"
        self.settings_path = self.directory / "settings.json"
        self.settings = read_state(self.settings_path)
        self.records = {}
        for path in (self.directory / "runs").glob("*/workflow.json"):
            record = read_state(path)
            if path.parent.name == record.get("id") and ID.fullmatch(path.parent.name):
                self.records[record["id"]] = record
        self.observed = {}
        self.ready = {}
        self.errors = {}
        self.lock = threading.RLock()
        self.halt = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.next_check = 0

    def begin(self):
        self.halt.clear()
        self.thread = threading.Thread(target=self.work, name="labelpilot-automation", daemon=True)
        self.thread.start()

    def shutdown(self):
        self.halt.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)

    def settings_for(self, project_id):
        project = self.store.project(project_id)
        with self.lock:
            stored = copy.deepcopy(self.settings.get(project_id))
        return stored if stored is not None else recipe_defaults(project)

    def configure(self, project_id, payload):
        recipe = normalize_recipe(self.store, self.store.project(project_id), payload)
        with self.lock:
            self.save_recipe(project_id, recipe)
        self.wake.set()
        return recipe

    def save_recipe(self, project_id, recipe):
        updated = {**self.settings, project_id: recipe}
        atomic_json(self.settings_path, updated)
        self.settings = updated
        self.ready.pop(project_id, None)
        self.next_check = 0

    def record_path(self, record):
        return self.directory / "runs" / record["id"]

    def save(self, record, **updates):
        with self.lock:
            record.update(updates)
            atomic_json(self.record_path(record) / "workflow.json", record)
            self.records[record["id"]] = copy.deepcopy(record)

    def view(self, project_id):
        settings = self.settings_for(project_id)
        with self.lock:
            records = sorted((row for row in self.records.values() if row["project"] == project_id),
                             key=lambda row: row["created_at"], reverse=True)
            public = [{key: value for key, value in row.items()
                       if key not in ("recipe", "expected_dataset", "trigger", "only_files", "pinned", "companion")}
                      for row in records[:20]]
            return {"recipe": settings, "runs": copy.deepcopy(public), "readiness": self.ready.get(project_id),
                    "watch_error": self.errors.get(project_id)}

    def enqueue(self, project_id, kind, batch="", *, trigger=None, configuration=None):
        if kind not in ("intake", "routing", "release") or not isinstance(batch, str):
            raise Problem("无效的工作流类型或批次", 422)
        recipe = self.settings_for(project_id)
        if configuration is not None:
            if not isinstance(configuration, dict):
                raise Problem("无效的工作流设置", 422)
            recipe = normalize_recipe(self.store, self.store.project(project_id), configuration)
        if kind == "release":
            get_blindset(self.store, project_id, recipe["release"]["blindset"])
        with self.lock:
            if any(row["project"] == project_id and row["kind"] == kind and row["status"] in IN_PROGRESS
                   for row in self.records.values()):
                raise Problem("这个工作流已有任务在运行", 409)
            if configuration is not None:
                self.save_recipe(project_id, recipe)
            run_id = uuid.uuid4().hex
            created = now()
            if kind == "intake":
                batch = f"收图 {created[:10]} · {run_id[:6]}"
            elif kind == "release":
                batch = recipe["release"]["batch"]
            record = {"id": run_id, "project": project_id, "kind": kind, "batch": batch, "recipe": recipe,
                      "created_at": created, "status": "queued", "phase": {"intake":"scan", "routing":"route", "release":"prepare"}[kind],
                      "trigger": trigger}
            self.save(record)
        self.wake.set()
        return {"id": run_id, "recipe": recipe}

    def get(self, project_id, run_id):
        with self.lock:
            record = self.records.get(run_id)
            if not record or record["project"] != project_id:
                raise Problem("工作流任务不存在", 404)
            return copy.deepcopy(record)

    def retry(self, project_id, run_id):
        with self.lock:
            record = self.get(project_id, run_id)
            if record["status"] not in ("failed", "partial"):
                raise Problem("当前任务不需要重试", 409)
            phase = record["phase"]
            if phase == "done":
                phase = "scan"
                record["only_files"] = [row["relative"] for row in record.get("unreadable", [])]
                record.pop("prediction_id", None)
                record.pop("routing_id", None)
            child = {"predict":"prediction_id", "route":"routing_id", "train":"training_id", "compare":"comparison_id"}.get(phase)
            if child:
                record.pop(child, None)
            if phase == "train" and "审核数据在工作流启动后发生变化" in record.get("error", ""):
                phase = "prepare"
            self.save(record, status="queued", phase=phase, error="", progress=None)
        self.wake.set()
        return {"id": run_id}

    def work(self):
        while not self.halt.is_set():
            self.tick()
            self.wake.wait(1)
            self.wake.clear()

    def tick(self):
        with self.lock:
            records = [copy.deepcopy(row) for row in self.records.values() if row["status"] in IN_PROGRESS]
        for record in records:
            if self.halt.is_set():
                return
            try:
                if record["kind"] == "release":
                    self.advance_release(record)
                else:
                    self.advance_intake(record)
            except Problem as exc:
                if exc.status == 409 and record["phase"] in ("predict", "route", "train", "compare"):
                    if record.get("message") != exc.message or record["status"] != "waiting":
                        self.save(record, status="waiting", message=exc.message)
                else:
                    self.save(record, status="failed", error=exc.message)
            except Exception as exc:
                self.save(record, status="failed", error=str(exc)[:400])
        if time.monotonic() >= self.next_check:
            self.next_check = time.monotonic() + 5
            self.schedule()

    def schedule(self):
        with self.lock:
            projects = list(self.settings)
        for project_id in projects:
            try:
                if self.halt.is_set():
                    return
                recipe = self.settings_for(project_id)
                intake, release = recipe["intake"], recipe["release"]
                if intake["watch"] and not self.active(project_id, "intake"):
                    stable, _ = self.incoming(project_id, intake, automatic=True)
                    if stable:
                        self.enqueue(project_id, "intake", trigger="watch")
                if release["enabled"] and not self.active(project_id, "release"):
                    rows = self.approved(project_id, release)
                    state = readiness(rows, release)
                    with self.lock:
                        self.ready[project_id] = state
                    if state["ready"]:
                        key = fingerprint([[(row["id"], row["version"]) for row in rows], release])
                        with self.lock:
                            seen = any(row["project"] == project_id and row.get("trigger") == key for row in self.records.values())
                        if not seen:
                            self.enqueue(project_id, "release", trigger=key)
                with self.lock:
                    self.errors.pop(project_id, None)
            except Problem as exc:
                with self.lock:
                    self.errors[project_id] = exc.message
            except Exception as exc:
                with self.lock:
                    self.errors[project_id] = str(exc)[:300]

    def active(self, project_id, kind):
        with self.lock:
            return any(row["project"] == project_id and row["kind"] == kind and row["status"] in IN_PROGRESS
                       for row in self.records.values())

    def incoming(self, project_id, intake, *, automatic=False):
        folder = source_folder(self.store.project(project_id), intake["folder"])
        observed = self.observed.setdefault((project_id, str(folder)), {})
        with self.lock:
            suppressed = {(row["relative"], *row["fingerprint"]) for run in self.records.values()
                          if run["project"] == project_id and automatic for row in run.get("unreadable", [])}
        return observe_images(self.store, project_id, folder, observed, suppressed)

    def approved(self, project_id, release):
        project = self.store.project(project_id)
        rows = self.store.listing(project_id, batch=release["batch"], status="reviewed",
                                  limit=len(project.items), include_documents=True)["items"]
        for row in rows:
            row["image_sha256"] = row["document"].get("labelpilot", {}).get("image_sha256") or row["image_sha256"]
        return reserve_blind_images(self.store, project_id, rows)

    def child(self, record, field, get, start):
        # Allocate the child ID before dispatch so a restart can find every launched task.
        if field not in record:
            self.save(record, **{field: uuid.uuid4().hex}, status="running", message="")
        task_id = record[field]
        result = get(record["project"], task_id)
        if result is None:
            result = start(task_id)
        progress = {key: result[key] for key in ("status", "done", "total", "failed", "skipped", "flagged") if key in result}
        progress.update({key: value for key, value in result.get("progress", {}).items() if key in ("epoch", "epochs", "done", "total")})
        if record.get("progress") != progress or record["status"] != "running":
            self.save(record, progress=progress, status="running", message="")
        if result["status"] in CHILD_ACTIVE:
            return None
        if result["status"] != "completed":
            message = result.get("error") or (result.get("errors") or [{}])[-1].get("error")
            self.save(record, status="failed", error=message or "后台步骤未完成，可重试")
            return None
        return result

    def advance_intake(self, record):
        project, batch, stage = record["project"], record["batch"], record["phase"]
        intake = record["recipe"]["intake"]
        limit = 10000
        if stage == "scan":
            retry_files = record.get("only_files")
            paths, waiting = self.incoming(project, intake, automatic=record.get("trigger") == "watch" and retry_files is None)
            if retry_files is not None:
                paths = [row for row in paths if row[0] in retry_files]
            if not paths and waiting:
                if record["status"] != "waiting":
                    self.save(record, status="waiting", message="正在等待图片写入完成")
                return
            expected = {item["relative"]: item for item in self.store.project(project).items}
            images, failed, unsettled = verify_incoming(paths[:limit], expected)
            summary = self.store.import_batch(project, batch, images)
            if not images and unsettled:
                return
            if not summary["included"]:
                self.save(record, **summary, unreadable=failed, status="partial" if failed else "completed", phase="done")
                return
            self.save(record, **summary, unreadable=failed, phase="curate", status="running", message="")
        elif stage == "curate":
            result = similar_frames(self.store, project, batch, intake["similarity"])
            excluded = set(result["excluded"])
            if intake["representatives"]:
                for group in result["groups"]:
                    excluded.update(item["id"] for item in group["members"] if item["id"] != group["representative"])
                self.store.save_similarity_selection(project, batch, sorted(excluded))
            self.save(record, phase="predict", similar=len(result["groups"]), deferred=len(excluded))
        elif stage == "predict":
            existing = self.jobs.lookup(project, record.get("prediction_id"))
            if existing is None and not self.store.listing(project, batch=batch, status="curated", limit=1)["filtered"]:
                self.save(record, phase="route", progress=None)
                return
            if "pinned" not in record:
                model = weight_path(self.store.project(project).config.get("model_path", ""))
                self.save(record, pinned={"path": str(model), "sha256": digest(model)})
            job = self.child(record, "prediction_id", self.jobs.lookup, lambda task: self.jobs.start(
                project, {"batch": batch, "limit": limit, "confidence": intake["confidence"], "image_size": 640, "curated": True},
                task_id=task, pinned_model=record["pinned"]))
            if job:
                self.save(record, phase="route", progress=None)
        elif stage == "route":
            self.advance_routing(record, intake, limit)

    def advance_routing(self, record, intake, limit):
        project, batch = record["project"], record["batch"]
        if "companion" not in record:
            path = weight_path(intake["companion"]) if intake["companion"] else None
            self.save(record, companion={"path": str(path), "sha256": digest(path)} if path else None)
        key = fingerprint([intake["confidence"], intake["low"], intake["rare"], record["companion"]])
        settings = {"batch": batch, "limit": limit, "route_key": key, "sample": intake["sample"], "rare": intake["rare"],
                    "low": intake["low"], "image_size": 640, "confidence": intake["confidence"]}
        try:
            job = self.child(record, "routing_id", self.jobs.lookup, lambda task: self.jobs.start(
                project, settings, routing=True, task_id=task, pinned_model=record["companion"]))
        except Problem as exc:
            if exc.message != "当前没有需要重新分流的候选图":
                raise
            job = {}
        if job is None:
            return
        samples = stratify_review(self.store, project, batch, intake["sample"])
        queued = self.store.listing(project, batch=batch, status="triage", limit=1)["filtered"]
        self.save(record, phase="done", status="partial" if record.get("unreadable") else "completed",
                  routed=queued, samples=samples, progress=None)

    def advance_release(self, record):
        project, phase = record["project"], record["phase"]
        recipe = record["recipe"]["release"]
        directory = self.record_path(record)
        baseline = directory / "baseline.pt"
        if phase == "prepare":
            with self.store.lock:
                plan, rows = training_selection(self.store, project, recipe)
                state = readiness(rows, recipe)
                if not state["ready"]:
                    detail = "、".join(f"{name} {count} 张" for name, count in state["missing"].items())
                    raise Problem(f"清洗并排除盲测后有 {state['images']} 张，要求至少 {state['required']} 张。覆盖不足 {detail or '无'}", 422)
                snapshot = dataset_identity(rows, plan.project.names)
            if not baseline.is_file():
                original = weight_path(recipe["baseline"] or self.store.project(project).config.get("model_path", ""))
                temporary = baseline.with_suffix(".tmp")
                shutil.copyfile(original, temporary)
                if digest(temporary) != digest(original):
                    raise Problem("复制基线时权重发生变化，请重试", 422)
                temporary.replace(baseline)
            current = self.approved(project, recipe)
            trigger = fingerprint([[(row["id"], row["version"]) for row in current], recipe])
            self.save(record, phase="train", status="running", expected_dataset=snapshot, reviewed=state["images"],
                      baseline_sha256=digest(baseline), trigger=record.get("trigger") or trigger)
        elif phase == "train":
            if not record.get("training_id") and digest(baseline) != record["baseline_sha256"]:
                raise Problem("工作流基线权重已变化，请恢复该权重", 422)
            job = self.child(record, "training_id", self.training.lookup, lambda task: self.training.start(
                project, recipe, task_id=task, starting_weights=baseline, expected_dataset=record["expected_dataset"]))
            if job:
                self.save(record, phase="compare", progress=None)
        elif phase == "compare":
            weights = self.training.artifact(project, record["training_id"], "weights")
            if not record.get("comparison_id") and digest(baseline) != record["baseline_sha256"]:
                raise Problem("工作流基线权重已变化，请恢复该权重", 422)
            payload = {"set_id": recipe["blindset"], "old_path": str(baseline), "new_path": str(weights),
                       "confidence": recipe["eval_confidence"], "match_iou": recipe["eval_iou"], "image_size": 640}
            job = self.child(record, "comparison_id", self.lookup_comparison, lambda task: self.comparisons.start(
                project, payload, task_id=task))
            if job:
                report = read_json(self.comparisons.report(project, job["id"]))
                gates = assess_gates(report, recipe["gates"])
                self.save(record, phase="done", status="passed" if all(row["passed"] for row in gates) else "rejected",
                          gates=gates, progress=None, candidate_sha256=report["models"]["new"]["sha256"])

    def lookup_comparison(self, project_id, task_id):
        try:
            return self.comparisons.get(project_id, task_id)
        except Problem as exc:
            if exc.status != 404:
                raise
            return None

    def activate(self, project_id, run_id):
        record = self.get(project_id, run_id)
        if record["kind"] != "release" or record["status"] != "passed":
            raise Problem("当前训练候选未通过验收", 409)
        path = self.training.artifact(project_id, record["training_id"], "weights")
        if digest(path) != record["candidate_sha256"]:
            raise Problem("权重已变化，请重新验收", 409)
        with self.jobs.lock:
            if any(row["status"] in ("queued", "running") for row in self.jobs.records.values()) or not self.jobs.inference.lock.acquire(blocking=False):
                raise Problem("模型任务运行期间请稍后切换", 409)
            try:
                project = self.store.update_model(project_id, path)
                self.save(record, activated_at=now())
            finally:
                self.jobs.inference.lock.release()
        return project
