import copy
import hashlib
import json
import threading
import uuid

from .duplicates import bounds, box_iou, duplicate_threshold, filter_duplicates
from .store import Problem, atomic_json, now, read_annotation_snapshot


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def box_targets(project, current, options):
    if project.task_type not in ("detect", "segment"):
        raise Problem("检测框生成轮廓仅用于目标检测和实例分割项目", 422)
    indices = options.get("indices")
    if indices is not None and (not isinstance(indices, list) or not indices or len(indices) > 1000
                                or any(type(index) is not int or index < 0 for index in indices)):
        raise Problem("请选择有效的检测框", 422)
    label = options.get("label")
    all_labels = options.get("all_labels", True)
    if type(all_labels) is not bool or label is not None and label not in project.names:
        raise Problem("请选择当前项目已有标签", 422)
    prompt = options.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt.strip()) > 160:
        raise Problem("对象描述最多 160 个字符", 422)
    shapes = current["document"]["shapes"]
    if indices is not None and any(index >= len(shapes) or shapes[index].get("shape_type") != "rectangle" for index in indices):
        raise Problem("检测框已变化，请重新选择", 409)
    targets = []
    for index, shape in enumerate(shapes):
        if shape.get("shape_type") != "rectangle" or indices is not None and index not in indices:
            continue
        if not all_labels and shape["label"] != label:
            continue
        description = prompt.strip() if shape["label"] == label and prompt.strip() else shape["label"].replace("_", " ")
        targets.append({"source_index": index, "source_key": signature(shape), "label": shape["label"],
                        "box": list(bounds(shape)), "prompt": description, "source_shape": copy.deepcopy(shape)})
    if len(targets) > 1000:
        raise Problem("每张图片最多处理 1000 个检测框", 422)
    return targets


def predict_contours(store, sam3, project_id, item_id, options, *, job_id=None):
    project = store.project(project_id)
    current = store.detail(project_id, item_id)
    if options.get("expected_version", current["version"]) != current["version"]:
        raise Problem("标注版本已变化，请重新载入", 409)
    store.validate(project, current["document"], current["width"], current["height"], "draft", False)
    targets = box_targets(project, current, options)
    if not targets:
        raise Problem("所选范围没有尚待生成轮廓的矩形框", 422)
    arguments = {"targets": [{key: target[key] for key in ("box", "prompt", "source_index")} for target in targets],
                 "threshold": options.get("threshold", 0.35)}
    if job_id:
        result = sam3.predict(project.item(item_id)["path"], arguments, owner_job_id=job_id)
    else:
        result = sam3.predict(project.item(item_id)["path"], arguments)
    if (result.get("width"), result.get("height")) != (current["width"], current["height"]):
        raise Problem("SAM3 返回尺寸与原图不一致，已拒绝轮廓候选", 422)
    matches = {}
    for candidate in result.get("candidates", []):
        source_index = candidate.get("source_index")
        target = next((target for target in targets if target["source_index"] == source_index), None)
        if target is None:
            raise Problem("SAM3 返回未知检测框的候选，已拒绝应用", 422)
        shape = {"label": target["label"], "shape_type": "polygon", "points": candidate.get("points")}
        store.validate(project, {**current["document"], "shapes": [shape]}, current["width"], current["height"], "draft", False)
        overlap = box_iou(bounds(shape), target["box"])
        score = candidate.get("score")
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise Problem("SAM3 返回无效置信度", 422)
        if overlap < 0.2:
            continue
        if source_index not in matches or (overlap, score) > matches[source_index][0]:
            matches[source_index] = ((overlap, score), {**candidate, **target, "box_iou": overlap})
    candidates = [value[1] for _, value in sorted(matches.items())]
    after = store.detail(project_id, item_id)
    if (after["version"] != current["version"] or after["image_sha256"] != current["image_sha256"]
            or signature(after["document"]) != signature(current["document"]) or after["source_fingerprint"] != current["source_fingerprint"]):
        raise Problem("推理期间图片或标注已变化，已拒绝轮廓候选", 409)
    return {"candidates": candidates, "base_version": current["version"], "image_sha256": current["image_sha256"],
            "snapshot": signature(current["document"]), "width": current["width"], "height": current["height"],
            "target_count": len(targets), "unmatched": len(targets) - len(candidates),
            "backend": sam3.config().get("backend", "pytorch"), "created_at": now()}


def save_contours(store, project_id, item_id, result):
    with store.lock:
        current = store.detail(project_id, item_id)
        if (current["version"] != result["base_version"] or current["image_sha256"] != result["image_sha256"]
                or signature(current["document"]) != result["snapshot"]):
            raise Problem("标注已变化，未保存过期轮廓候选", 409)
        with store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT version FROM annotations WHERE project=? AND item=?", (project_id, item_id)).fetchone()
            if (row["version"] if row else 0) != result["base_version"]:
                raise Problem("标注已在其他进程更新", 409)
            connection.execute("INSERT OR REPLACE INTO contour_suggestions VALUES (?, ?, ?, ?, ?, ?)",
                               (project_id, item_id, current["version"], result["snapshot"], current["image_sha256"],
                                json.dumps(result, ensure_ascii=False, allow_nan=False)))


def load_contours(store, project_id, item_id):
    current = store.detail(project_id, item_id)
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM contour_suggestions WHERE project=? AND item=?", (project_id, item_id)).fetchone()
    if row is None:
        return {"candidates": [], "stale": False}
    if (row["version"] != current["version"] or row["image_sha256"] != current["image_sha256"]
            or row["snapshot"] != signature(current["document"])):
        return {"candidates": [], "stale": True}
    return json.loads(row["result"])


class AssistanceJobs:
    def __init__(self, store, jobs, sam3):
        self.store, self.jobs, self.sam3 = store, jobs, sam3

    def start(self, project_id, options, *, cleanup=False):
        project = self.store.project(project_id)
        if project.task_type not in ("detect", "segment"):
            raise Problem("此功能用于目标检测和实例分割项目", 422)
        limit = options.get("limit", 100)
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise Problem("每批数量必须为 1 到 10000", 422)
        if any(not isinstance(options.get(key, ""), str) for key in ("query", "batch", "status")):
            raise Problem("队列筛选必须为文本", 422)
        if cleanup:
            duplicate_threshold(options)
        else:
            if options.get("indices") is not None:
                raise Problem("批量轮廓任务不接受单张图片的框序号", 422)
            if type(options.get("regenerate", False)) is not bool:
                raise Problem("重新生成开关必须为布尔值", 422)
            box_targets(project, {"document": {"shapes": []}}, options)
            threshold = options.get("threshold", .35)
            if type(threshold) not in (int, float) or not .05 <= threshold <= .95:
                raise Problem("置信度阈值必须在 0.05 到 0.95 之间", 422)
        listing = self.store.listing(project_id, query=options.get("query", ""), batch=options.get("batch", ""),
                                     status="predicted" if cleanup else options.get("status", ""), limit=1000000,
                                     include_documents=not cleanup)
        cached = {}
        if not cleanup and not options.get("regenerate", False):
            with self.store.connect() as connection:
                cached = {row["item"]: (row["version"], row["snapshot"]) for row in connection.execute(
                    "SELECT item,version,snapshot FROM contour_suggestions WHERE project=?", (project_id,))}
        ids = []
        for item in listing["items"]:
            if not item.get("count", 0):
                continue
            if not cleanup:
                document = item.get("document")
                if document is None:
                    source = project.item(item["id"])["source"]
                    document = read_annotation_snapshot(source)[0] if source else {"shapes": []}
                document = project.normalize_labels(document)
                if not box_targets(project, {"document": document}, options):
                    continue
                if cached.get(item["id"]) == (item.get("version", 0), signature(document)):
                    continue
            ids.append(item["id"])
            if len(ids) == limit:
                break
        if not ids:
            raise Problem("当前范围没有模型待审核标注" if cleanup else "当前范围没有待生成轮廓的矩形框图片；已有候选可直接读取，或勾选重新生成", 422)
        job_id = uuid.uuid4().hex
        with self.jobs.lock:
            if any(record["status"] in ("queued", "running") for record in self.jobs.records.values()):
                raise Problem("已有批量任务运行中，请先完成或取消", 409)
            if not cleanup and self.jobs.inference.exclusive.is_set() and not self.sam3.slot_reserved:
                raise Problem("模型运行环境正在使用", 409)
            record = {"id": job_id, "project": project_id, "kind": "duplicate_cleanup" if cleanup else "sam3_contours",
                      "status": "queued", "total": len(ids), "done": 0, "skipped": 0, "failed": 0,
                      "removed": 0, "candidates": 0, "unmatched": 0, "errors": [], "created_at": now(), "options": copy.deepcopy(options)}
            self.jobs.records[job_id] = record
            self.jobs.cancel_events[job_id] = threading.Event()
            atomic_json(self.jobs.directory / f"{job_id}.json", record)
        threading.Thread(target=self.run, args=(job_id, ids), daemon=True).start()
        return copy.deepcopy(record)

    def run(self, job_id, ids):
        record = self.jobs.records[job_id]
        project_id, options = record["project"], record["options"]
        cleanup = record["kind"] == "duplicate_cleanup"
        event = self.jobs.cancel_events[job_id]
        self.jobs.update(job_id, status="running")
        progress = dict.fromkeys(("done", "skipped", "failed", "removed", "candidates", "unmatched"), 0)
        errors = []
        for item_id in ids:
            if event.is_set():
                break
            try:
                current = self.store.detail(project_id, item_id)
                if cleanup:
                    if current["status"] != "predicted":
                        progress["skipped"] += 1
                        continue
                    shapes = current["document"]["shapes"]
                    predicted = [shape for shape in shapes if shape.get("prediction_origin") == "model_prediction"]
                    protected = [shape for shape in shapes if shape.get("prediction_origin") != "model_prediction"]
                    filtered, removed = filter_duplicates(predicted, duplicate_threshold(options), protected=protected)
                    if removed and not event.is_set():
                        keep_ids = {id(shape) for shape in [*protected, *filtered]}
                        document = copy.deepcopy(current["document"])
                        document["shapes"] = [copy.deepcopy(shape) for shape in shapes if id(shape) in keep_ids]
                        document.setdefault("labelpilot", {})["duplicate_cleanup"] = {"removed": len(removed), "iou": duplicate_threshold(options), "created_at": now()}
                        self.store.save(project_id, item_id, {**current, "document": document, "expected_version": current["version"], "status": "draft"})
                        progress["removed"] += len(removed)
                else:
                    if not box_targets(self.store.project(project_id), current, options):
                        progress["skipped"] += 1
                        continue
                    result = predict_contours(self.store, self.sam3, project_id, item_id, options, job_id=job_id)
                    if event.is_set():
                        break
                    save_contours(self.store, project_id, item_id, result)
                    progress["candidates"] += len(result["candidates"])
                    progress["unmatched"] += result["unmatched"]
                progress["done"] += 1
            except Problem as error:
                progress["skipped" if error.status == 409 else "failed"] += 1
                errors.append({"item": item_id, "error": error.message})
            except Exception as error:
                progress["failed"] += 1
                errors.append({"item": item_id, "error": str(error)[:300]})
            finally:
                self.jobs.update(job_id, **progress, errors=errors[-50:])
            if progress["failed"] >= 3 and not progress["done"]:
                self.jobs.update(job_id, status="failed", finished_at=now())
                return
        self.jobs.update(job_id, status="cancelled" if event.is_set() else "completed_with_errors" if progress["failed"] else "completed", finished_at=now())
