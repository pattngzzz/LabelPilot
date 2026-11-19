import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath

from PIL import Image

from .evaluation import iou
from .store import Problem, source_group


FRAME_NAME = re.compile(r"^(.*?)([0-9]+)$")
PROPAGATION_TARGETS = ("unstarted", "draft", "imported", "predicted")


def box_bounds(shape):
    xs, ys = zip(*shape["points"])
    return min(xs), min(ys), max(xs), max(ys)


def filename_frame(item):
    path = PurePosixPath(item["relative"])
    match = FRAME_NAME.fullmatch(path.stem)
    return (path.parent, match[1].casefold(), int(match[2])) if match else None


def sequence_frames(project, start, *, require_index=False):
    group = source_group(start["source_group"])
    folder = PurePosixPath(start["relative"]).parent
    named = filename_frame(start)
    if group:
        items = [item for item in project.items if item["source_group"] == group]
    else:
        items = [item for item in project.items if not source_group(item["source_group"])
                 and PurePosixPath(item["relative"]).parent == folder]
        if named:
            items = [item for item in items if (frame := filename_frame(item)) and frame[:2] == named[:2]]
    indices = [item["frame_index"] for item in items]
    if all(index is not None for index in indices) and len(set(indices)) == len(indices):
        return sorted(zip(indices, items), key=lambda row: row[0])
    indexed_names = [filename_frame(item) for item in items]
    if (all(indexed_names) and len({frame[:2] for frame in indexed_names}) == 1
            and len({frame[2] for frame in indexed_names}) == len(items)):
        return sorted(zip((frame[2] for frame in indexed_names), items), key=lambda row: row[0])
    if require_index:
        raise Problem("关键帧需要明确的帧序，请在任务清单提供 frame_index，或使用同前缀的编号文件名", 422)
    return list(enumerate(items))


def read_keyframe(store, project, specification):
    if (not isinstance(specification, dict) or not isinstance(specification.get("id"), str)
            or type(specification.get("version")) is not int):
        raise Problem("请先选择一个矩形框作为关键帧", 422)
    current = store.detail(project.id, specification["id"])
    if (specification.get("version"), specification.get("image_sha256"), specification.get("source_fingerprint")) != (
        current["version"], current["image_sha256"], current["source_fingerprint"]
    ):
        raise Problem("关键帧标注已更新，请重新选框", 409)
    index = specification.get("shape")
    shapes = current["document"]["shapes"]
    if type(index) is not int or not 0 <= index < len(shapes) or shapes[index].get("shape_type") != "rectangle":
        raise Problem("请先选中一个矩形框", 422)
    shape = shapes[index]
    store.validate(project, {**current["document"], "shapes": [shape]}, current["width"], current["height"], "draft", False)
    return current, shape


def normalized_box(current, shape):
    return tuple(value / size for value, size in zip(box_bounds(shape), (current["width"], current["height"]) * 2))


def propagated_draft(current, anchor_shape, relative_box, origin, anchor, end_id=None):
    label = anchor_shape["label"]
    box = [value * size for value, size in zip(relative_box, (current["width"], current["height"]) * 2)]
    for existing in current["document"]["shapes"]:
        if existing.get("label") == label and existing.get("shape_type") == "rectangle":
            same_anchor = (existing.get("source_item_id"), existing.get("source_shape"), existing.get("source_revision")) == (
                anchor["id"], anchor["shape"], anchor["version"]
            )
            if same_anchor or iou(box, box_bounds(existing)) >= .9:
                return None
    shape = {key: value for key, value in anchor_shape.items() if key not in ("score", "prediction_id", "prediction_origin")}
    shape.update(points=[box[:2], box[2:]], annotation_origin=origin, source_item_id=anchor["id"],
                 source_shape=anchor["shape"], source_revision=anchor["version"])
    if end_id is not None:
        shape["end_item_id"] = end_id
    else:
        shape.pop("end_item_id", None)
    document = {**current["document"], "shapes": [*current["document"]["shapes"], shape]}
    payload = {**current, "document": document, "expected_version": current["version"], "status": "draft"}
    return payload, box


def inherit_next(store, project_id, specification):
    project = store.project(project_id)
    with store.lock:
        current, shape = read_keyframe(store, project, specification)
        frames = sequence_frames(project, current["item"])
        index = next(index for index, (_, item) in enumerate(frames) if item["id"] == current["item"]["id"])
        if index == len(frames) - 1:
            raise Problem("已到同一来源的最后一张图片", 422)
        destination = frames[index + 1][1]["id"]
        target = store.detail(project_id, destination)
        if target["status"] not in PROPAGATION_TARGETS:
            raise Problem("下一张已审核或跳过，请使用其他未审核帧", 409)
        store.validate(project, target["document"], target["width"], target["height"], "draft", False)
        draft = propagated_draft(target, shape, normalized_box(current, shape), "inherit_previous", specification)
        if draft:
            store.save(project_id, destination, draft[0])
        return {"id": destination, "appended": bool(draft)}


def interpolate_frames(store, project_id, payload, *, apply=False):
    project = store.project(project_id)
    with store.lock:
        start, first = read_keyframe(store, project, payload.get("start"))
        end, last = read_keyframe(store, project, payload.get("end"))
        start_id, end_id = start["item"]["id"], end["item"]["id"]
        if first["label"] != last["label"]:
            raise Problem("起止关键帧需要选择同一个目标的同类框", 422)
        frames = sequence_frames(project, start["item"], require_index=True)
        positions = {item["id"]: position for position, item in frames}
        if end_id not in positions or positions[end_id] <= positions[start_id]:
            raise Problem("终点需位于同一来源且晚于起点，请选择最后一帧可见目标", 422)
        start_index, end_index = positions[start_id], positions[end_id]
        first_box, last_box = normalized_box(start, first), normalized_box(end, last)
        entries, targets, dependencies = [], [], []
        skipped = {"status": 0, "matched": 0, "unreadable": 0}
        for position, item in frames:
            if not start_index < position < end_index:
                continue
            try:
                current = store.detail(project_id, item["id"])
                if current["status"] in PROPAGATION_TARGETS:
                    store.validate(project, current["document"], current["width"], current["height"], "draft", False)
            except (Problem, OSError, ValueError):
                skipped["unreadable"] += 1
                dependencies.append((item["id"], "unreadable"))
                continue
            dependencies.append((item["id"], current["status"], current["version"], current["image_sha256"], current["source_fingerprint"]))
            if current["status"] not in PROPAGATION_TARGETS:
                skipped["status"] += 1
                continue
            fraction = (position - start_index) / (end_index - start_index)
            relative = [start_value + fraction * (end_value - start_value) for start_value, end_value in zip(first_box, last_box)]
            draft = propagated_draft(current, first, relative, "keyframe_interpolation", payload["start"], end_id)
            if draft is None:
                skipped["matched"] += 1
                continue
            document, box = draft
            entries.append((item["id"], document))
            targets.append({"id": item["id"], "photo_id": item["photo_id"], "width": current["width"],
                            "height": current["height"], "box": box})
        token_data = [payload["start"], payload["end"], first_box, last_box, dependencies, targets]
        token = hashlib.sha256(json.dumps(token_data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if apply:
            if payload.get("token") != token:
                raise Problem("区间里的图片或标注已变化，请重新预览", 409)
            if entries:
                store.save_many(project_id, entries)
        return {"token": token, "targets": targets, "skipped": skipped, "label": first["label"]}


def prompt_points(value, width, height):
    if not isinstance(value, list) or len(value) > 24:
        raise Problem("前景和背景分别最多 24 个提示点", 422)
    for point in value:
        if (not isinstance(point, list) or len(point) != 2
                or any(type(coord) not in (int, float) or not math.isfinite(coord) for coord in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)):
            raise Problem("提示点需位于原图内", 422)
    return value


def prompt_box(store, project_id, item_id, payload):
    current = store.detail(project_id, item_id)
    if payload.get("image_sha256") != current["image_sha256"]:
        raise Problem("原图已变化，请重新载入", 409)
    width, height = current["width"], current["height"]
    positive = prompt_points(payload.get("positive", []), width, height)
    negative = prompt_points(payload.get("negative", []), width, height)
    if not positive:
        raise Problem("请先点击目标内的前景点", 422)
    path = Path(store.project(project_id).item(item_id)["path"])
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != current["image_sha256"]:
        raise Problem("原图已变化，请重新载入", 409)
    with Image.open(io.BytesIO(contents)) as original:
        image = original.convert("RGB")
        image.thumbnail((640, 640), Image.Resampling.BILINEAR)
    box = segment_prompt(image, positive, negative, (width, height))
    return {"box": box, "image_sha256": current["image_sha256"]}


def segment_prompt(image, positive, negative, original_size):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise Problem("点选辅助需要安装 LabelPilot 的 OpenCV 依赖", 422) from exc
    width, height = image.size
    if min(width, height) < 4:
        raise Problem("图片尺寸过小，无法生成点选候选", 422)

    def scaled(points):
        return [(min(width - 1, int(x * width / original_size[0])),
                 min(height - 1, int(y * height / original_size[1]))) for x, y in points]

    inside, outside = scaled(positive), scaled(negative)
    if set(inside) & set(outside):
        raise Problem("前景点与背景点太近，请移动其中一个提示点", 422)
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    distance = np.full((height, width), np.inf, dtype=np.float32)
    for x, y in inside:
        np.minimum(distance, np.sum((lab - lab[y, x]) ** 2, axis=2), out=distance)
    _, regions = cv2.connectedComponents((distance <= 28**2).astype(np.uint8), connectivity=8)
    probable = np.isin(regions, [regions[y, x] for x, y in inside])
    mask = np.where(probable, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[[0, -1], :] = cv2.GC_BGD
    mask[:, [0, -1]] = cv2.GC_BGD
    for (x, y), marker in [(point, cv2.GC_BGD) for point in outside] + [(point, cv2.GC_FGD) for point in inside]:
        mask[y, x] = marker
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None,
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise Problem("当前提示点无法分离前景，请补充目标内外的点", 422) from exc
    _, segments = cv2.connectedComponents((mask % 2).astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(np.isin(segments, [segments[y, x] for x, y in inside]))
    if len(xs) < 4 or xs.min() == xs.max() or ys.min() == ys.max() or len(xs) > width * height * .98:
        raise Problem("尚未得到清晰的前景轮廓，请补充目标内或背景提示点", 422)
    x_scale, y_scale = original_size[0] / width, original_size[1] / height
    return [float(xs.min()) * x_scale, float(ys.min()) * y_scale,
            float(xs.max() + 1) * x_scale, float(ys.max() + 1) * y_scale]
