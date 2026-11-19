import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

from .annotation_tools import PROPAGATION_TARGETS, normalized_box, propagated_draft, read_keyframe, sequence_frames
from .store import Problem


def track_step(before, after, box):
    import cv2
    import numpy as np

    height, width = before.shape
    left, top, right, bottom = box
    x1, y1, x2, y2 = int(left), int(top), math.ceil(right), math.ceil(bottom)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, "目标太小，请手动选择新起点"
    mask = np.zeros(before.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(before, maxCorners=100, qualityLevel=.008, minDistance=3, mask=mask)
    if points is None or len(points) < 4:
        return None, "框内可跟踪的纹理不足"

    next_points, found, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None)
    if next_points is None or found is None:
        return None, "目标跟踪中断"
    back, returned, _ = cv2.calcOpticalFlowPyrLK(after, before, next_points, None)
    if back is None or returned is None:
        return None, "目标跟踪不稳，需判断遮挡或出画"
    consistent = ((found.ravel() == 1) & (returned.ravel() == 1) &
                  (np.linalg.norm(back - points, axis=2).ravel() <= 1.5))
    if consistent.sum() < max(4, math.ceil(len(points) * .55)):
        return None, "目标跟踪不稳，需判断遮挡或出画"

    transform, inliers = cv2.estimateAffinePartial2D(points[consistent], next_points[consistent],
                                                     method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if transform is None or inliers.mean() < .65:
        return None, "目标位移不一致，需人工修正"
    scale = math.hypot(transform[0, 0], transform[1, 0])
    if not .65 <= scale <= 1.5:
        return None, "目标尺寸变化过大"
    corners = np.array([[left, top], [right, top], [left, bottom], [right, bottom]])
    moved = corners @ transform[:, :2].T + transform[:, 2]
    minimum, maximum = moved.min(axis=0), moved.max(axis=0)
    if (minimum < [-1, -1]).any() or (maximum > [width + 1, height + 1]).any():
        return None, "目标到达画面边缘，需确认是否出画"
    x1, y1, x2, y2 = (float(value) for value in (*minimum, *maximum))
    moved_box = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    new_x, new_y = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(new_x - (left + right) / 2) > width * .28 or abs(new_y - (top + bottom) / 2) > height * .28:
        return None, "相邻帧跳变过大"

    def patch(image, rect):
        x1, y1, x2, y2 = rect
        cropped = image[int(y1):math.ceil(y2), int(x1):math.ceil(x2)]
        return cv2.resize(cropped, (24, 24)).astype(np.float32)

    if moved_box[2] - moved_box[0] < 3 or moved_box[3] - moved_box[1] < 3:
        return None, "目标可能已出画"
    if np.abs(patch(before, box) - patch(after, moved_box)).mean() > 27:
        return None, "目标外观变化过大，需确认遮挡"
    return moved_box, None


def load_tracking_frame(project, current, size):
    import numpy as np

    path = Path(project.item(current["item"]["id"])["path"])
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != current["image_sha256"]:
        raise Problem("原图在跟踪期间发生变化", 409)
    with Image.open(io.BytesIO(contents)) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR))


def track_frames(store, project_id, request, *, apply=False):
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise Problem("跟踪辅助需要 OpenCV", 422) from exc
    project = store.project(project_id)
    anchor = request.get("start")
    current, shape = read_keyframe(store, project, anchor)
    count = request.get("limit", 40)
    if type(count) is not int or not 1 <= count <= 200:
        raise Problem("本段跟踪长度需为 1 到 200 帧", 422)
    frame_width, frame_height = current["width"], current["height"]
    divisor = max(1, max(frame_width, frame_height) / 720)
    size = max(1, round(frame_width / divisor)), max(1, round(frame_height / divisor))
    if min(size) < 8:
        raise Problem("图像太小，无法跟踪", 422)
    frames = sequence_frames(project, current["item"], require_index=True)
    start = next(i for i, (_, item) in enumerate(frames) if item["id"] == anchor["id"])
    reference = load_tracking_frame(project, current, size)
    box = tuple(value * dimension for value, dimension in zip(normalized_box(current, shape), size * 2))
    entries, targets = [], []
    skipped, stop = 0, None
    for _, item in frames[start + 1:start + 1 + count]:
        try:
            next_frame = store.detail(project_id, item["id"])
            if abs(next_frame["width"] * frame_height / (next_frame["height"] * frame_width) - 1) > .02:
                stop = {"id": item["id"], "photo_id": item["photo_id"], "reason": "图像比例改变，需确认场景切换"}
                break
            candidate = load_tracking_frame(project, next_frame, size)
        except (OSError, ValueError, Problem) as exc:
            stop = {"id": item["id"], "photo_id": item["photo_id"], "reason": getattr(exc, "message", "无法读取当前帧")}
            break
        moved, reason = track_step(reference, candidate, box)
        if moved is None:
            stop = {"id": item["id"], "photo_id": item["photo_id"], "reason": reason}
            break
        reference, box = candidate, moved
        if next_frame["status"] not in PROPAGATION_TARGETS:
            skipped += 1
            continue
        project_doc = next_frame["document"]
        store.validate(project, project_doc, next_frame["width"], next_frame["height"], "draft", False)
        relative = [value / dimension for value, dimension in zip(box, size * 2)]
        draft = propagated_draft(next_frame, shape, relative, "optical_flow_track", anchor)
        if draft is None:
            skipped += 1
            continue
        payload, image_box = draft
        entries.append((item["id"], payload))
        targets.append({"id": item["id"], "photo_id": item["photo_id"], "width": next_frame["width"],
                        "height": next_frame["height"], "box": image_box})
    stamps = [(item_id, payload["version"], payload["image_sha256"], payload["source_fingerprint"]) for item_id, payload in entries]
    token = hashlib.sha256(json.dumps([anchor, count, stamps, targets, stop], sort_keys=True).encode()).hexdigest()
    if apply:
        if request.get("token") != token:
            raise Problem("跟踪结果发生变化，请重新预览", 409)
        with store.lock:
            read_keyframe(store, project, anchor)
            if entries:
                store.save_many(project_id, entries)
    return {"token": token, "targets": targets, "skipped": skipped, "stop": stop, "label": shape["label"]}
