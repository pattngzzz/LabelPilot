import hashlib
import io
import math
import time
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from PIL import Image

from .annotation_tools import box_bounds
from .evaluation import count_detections
from .store import IMAGE_SUFFIXES, Problem, source_group


def source_folder(project, value):
    if not isinstance(value, str):
        raise Problem("监控目录必须为本机路径", 422)
    try:
        path = (project.root / value.strip()).resolve() if value.strip() else project.root
    except (OSError, ValueError, RuntimeError) as exc:
        raise Problem("无法解析收图目录", 422) from exc
    if not path.is_dir() or not path.is_relative_to(project.root):
        raise Problem("请选择当前项目的原图目录或其子目录", 422)
    return path


def image_state(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def observe_images(store, project_id, folder, observations, suppressed=()):
    project = store.project(project_id)
    receipts = store.intake_receipts(project_id)
    overlays = store.overlay(project_id)
    known = set(receipts)
    known.update(item["relative"] for item in project.items if overlays.get(item["id"], item)["status"] != "unstarted")
    stable, waiting, current = [], 0, {}
    moment = time.monotonic()
    for entry in sorted(folder.rglob("*")):
        if entry.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            path = entry.resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                continue
            relative = path.relative_to(project.root).as_posix()
            if relative in known or relative in current:
                continue
            fingerprint = image_state(path)
        except (OSError, RuntimeError):
            continue
        previous, since = observations.get(relative, (None, moment))
        since = since if previous == fingerprint else moment
        current[relative] = (fingerprint, since)
        if (relative, *fingerprint) in suppressed:
            continue
        if moment - since >= 1:
            stable.append((relative, path, fingerprint))
        else:
            waiting += 1
    observations.clear()
    observations.update(current)
    return stable, waiting


def verify_incoming(stable, expected):
    images, failed, unsettled = [], [], 0
    for relative, path, fingerprint in stable:
        try:
            contents = path.read_bytes()
            if image_state(path) != fingerprint:
                unsettled += 1
                continue
            with Image.open(io.BytesIO(contents)) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("请先规范化图片方向")
                image.load()
                width, height = image.size
            sha256 = hashlib.sha256(contents).hexdigest()
            original = expected.get(relative, {})
            if original.get("image_sha256") and original["image_sha256"] != sha256:
                raise ValueError("图片与原任务清单的指纹不符")
            if original.get("width") and (original["width"], original["height"]) != (width, height):
                raise ValueError("图片与原任务清单的尺寸不符")
            images.append({"relative": relative, "width": width, "height": height, "sha256": sha256})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            failed.append({"relative": relative, "fingerprint": list(fingerprint), "error": str(exc)[:130]})
    return images, failed, unsettled


def rare_labels(store, project_id, minimum):
    counts = Counter()
    for row in store.overlay(project_id, include_documents=True).values():
        if row["status"] == "reviewed":
            counts.update(shape["label"] for shape in row["document"]["shapes"])
    return {name for name in store.project(project_id).names if counts[name] < minimum}


def models_disagree(classes, primary, secondary):
    counts = {name: dict.fromkeys(("ground_truth", "matched", "missed", "false_positives"), 0) for name in classes}
    labels = [(shape["label"], box_bounds(shape)) for shape in primary]
    candidates = [(shape["label"], shape["score"], box_bounds(shape)) for shape in secondary]
    count_detections(counts, labels, candidates, .5)
    return any(row["missed"] or row["false_positives"] for row in counts.values())


def route_review(store, project_id, item_id, current, prediction, rare, low, key):
    shapes = current["document"]["shapes"]
    names = store.project(project_id).names
    store.validate(store.project(project_id), current["document"], current["width"], current["height"], "draft", False)
    empty = not shapes
    low_score = any(type(shape.get("score")) in (float, int) and shape["score"] < low for shape in shapes)
    sparse = sorted({shape["label"] for shape in shapes if shape["label"] in rare})
    disagree = prediction is not None and models_disagree(names, shapes, prediction["shapes"])
    reasons = (["empty"] if empty else []) + (["disagree"] if disagree else [])
    reasons += (["rare"] if sparse else []) + (["low"] if low_score else [])
    priority = 40 * empty + 30 * disagree + 15 * bool(sparse) + 10 * low_score
    with store.lock:
        latest = store.detail(project_id, item_id)
        if any(latest[key] != current[key] for key in ("version", "image_sha256", "source_fingerprint")):
            raise Problem("标注已更新，请重新分流", 409)
        if prediction and (prediction["base_version"], prediction["image_sha256"]) != (current["version"], current["image_sha256"]):
            raise Problem("第二模型推理时标注或图片已更新", 409)
        store.save_review_route(project_id, item_id, current["version"],
                                {"reasons": reasons, "rare": sparse, "priority": priority,
                                 "secondary_checked": prediction is not None, "key": key})
    return bool(reasons)


def stratify_review(store, project_id, batch, percent):
    rows = store.listing(project_id, batch=batch, status="triage", limit=len(store.project(project_id).items))["items"]
    groups = defaultdict(list)
    for row in rows:
        if not row["route"]["priority"]:
            group = source_group(row["source_group"]) or str(PurePosixPath(row["relative"]).parent)
            groups[group].append(row["id"])
    samples = set()
    for group, ids in groups.items():
        ordered = sorted(ids, key=lambda item_id: hashlib.sha256(f"{group}:{item_id}".encode()).digest())
        samples.update(ordered[:math.ceil(len(ordered) * percent / 100)])
    store.set_review_sample(project_id, [row["id"] for row in rows], samples)
    return len(samples)
