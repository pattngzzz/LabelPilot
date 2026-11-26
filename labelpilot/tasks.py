import base64
import binascii
import io
import math

from PIL import Image, ImageDraw


SHAPE_TYPES = {
    "ocr": ("rectangle", "polygon"),
    "detect": ("rectangle", "polygon"),
    "segment": ("rectangle", "polygon"),
    "semantic": ("polygon", "brush", "mask"),
    "hand_pose": ("hand",),
}
EXPORT_FORMATS = {
    "ocr": ("paddleocr", "labelme"),
    "detect": ("yolo", "labelme", "coco"),
    "segment": ("yolo", "labelme", "coco"),
    "semantic": ("mask",),
    "hand_pose": ("yolo", "coco"),
}
MODEL_TASK = {"detect": "detect", "segment": "segment", "semantic": "semantic", "hand_pose": "pose"}
TRAIN_MODEL = {
    "detect": "yolo11n.yaml",
    "segment": "yolo11n-seg.yaml",
    "semantic": "yolo26n-sem.yaml",
    "hand_pose": "yolo11n-pose.yaml",
}
BACKGROUND = "background"
MAX_MASK_PIXELS = 32_000_000
HAND_KEYPOINTS = (
    ("wrist", "手腕"),
    ("thumb_cmc", "拇指根部"), ("thumb_mcp", "拇指掌指关节"), ("thumb_ip", "拇指指间关节"), ("thumb_tip", "拇指尖"),
    ("index_mcp", "食指根部"), ("index_pip", "食指第一关节"), ("index_dip", "食指末端关节"), ("index_tip", "食指尖"),
    ("middle_mcp", "中指根部"), ("middle_pip", "中指第一关节"), ("middle_dip", "中指末端关节"), ("middle_tip", "中指尖"),
    ("ring_mcp", "无名指根部"), ("ring_pip", "无名指第一关节"), ("ring_dip", "无名指末端关节"), ("ring_tip", "无名指尖"),
    ("pinky_mcp", "小指根部"), ("pinky_pip", "小指第一关节"), ("pinky_dip", "小指末端关节"), ("pinky_tip", "小指尖"),
)
HAND_COUNT = len(HAND_KEYPOINTS)
HAND_SKELETON = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)) + tuple(
    (start + offset, start + offset + 1) for start in (1, 5, 9, 13, 17) for offset in range(3)
)


def model_labels(task, labels):
    return [BACKGROUND, *labels] if task == "semantic" else list(labels)


def mapped_model_names(names, project_names, label_aliases=None, model_label_aliases=None):
    aliases = label_aliases or {}
    model_aliases = model_label_aliases or {}
    ordered = []
    for index in range(len(names)):
        name = names[index]
        if index < len(project_names) and name == project_names[index]:
            ordered.append(name)
        elif index < len(project_names):
            ordered.append(model_aliases.get(name, aliases.get(name, name)))
        else:
            ordered.append(aliases.get(name, name))
    return ordered


def polygon_area(points):
    return abs(sum(x * following[1] - following[0] * y
                   for (x, y), following in zip(points, points[1:] + points[:1]))) / 2


def is_rotated_box(shape):
    angle, points = shape.get("rotation"), shape.get("points")
    if (shape.get("shape_type") not in ("rectangle", "polygon") or type(angle) not in (int, float) or not math.isfinite(angle)
            or not isinstance(points, list) or len(points) != 4):
        return False
    if any(not isinstance(point, list) or len(point) != 2
           or any(type(value) not in (int, float) or not math.isfinite(value) for value in point) for point in points):
        return False
    turns = [(following[0] - current[0]) * (after[1] - following[1])
             - (following[1] - current[1]) * (after[0] - following[0])
             for current, following, after in zip(points, points[1:] + points[:1], points[2:] + points[:2])]
    return all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)


def rectangle_corners(points):
    if len(points) != 4:
        return False
    edges = [(following[0] - point[0], following[1] - point[1])
             for point, following in zip(points, points[1:] + points[:1])]
    lengths = [math.hypot(*edge) for edge in edges]
    if min(lengths) < 1e-8:
        return False
    for index, edge in enumerate(edges):
        following, opposite = edges[(index + 1) % 4], edges[(index + 2) % 4]
        if (abs(edge[0] * following[0] + edge[1] * following[1]) > 1e-7 * lengths[index] * lengths[(index + 1) % 4]
                or math.hypot(edge[0] + opposite[0], edge[1] + opposite[1]) > 1e-7 * max(lengths[index], lengths[(index + 2) % 4])):
            return False
    return True


def simple_polygon(points):
    if len(points) < 3 or len({tuple(point) for point in points}) != len(points) or polygon_area(points) < 5e-9:
        return False

    def turn(first, second, third):
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

    def on_edge(first, second, point):
        return (abs(turn(first, second, point)) < 1e-9
                and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
                and min(first[1], second[1]) <= point[1] <= max(first[1], second[1]))

    edges = list(zip(points, points[1:] + points[:1]))
    for index, (first, second) in enumerate(edges):
        previous = points[index - 1]
        if abs(turn(previous, first, second)) < 1e-9 and ((first[0] - previous[0]) * (second[0] - first[0])
                                                       + (first[1] - previous[1]) * (second[1] - first[1])) < 0:
            return False
        for following in range(index + 1, len(edges)):
            if following == index + 1 or index == 0 and following == len(edges) - 1:
                continue
            third, fourth = edges[following]
            if ((turn(first, second, third) * turn(first, second, fourth) < 0
                 and turn(third, fourth, first) * turn(third, fourth, second) < 0)
                    or on_edge(first, second, third) or on_edge(first, second, fourth)
                    or on_edge(third, fourth, first) or on_edge(third, fourth, second)):
                return False
    return True


def validate_hand(shape, bounds):
    keypoints = shape.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != HAND_COUNT:
        raise ValueError("手部姿态需要 21 个固定顺序的关键点")
    left, top, right, bottom = bounds
    for keypoint in keypoints:
        if not isinstance(keypoint, list) or len(keypoint) != 3:
            raise ValueError("手部关键点必须是 [x, y, 可见性]")
        x, y, visible = keypoint
        if type(visible) is not int or visible not in (0, 1, 2):
            raise ValueError("可见性必须为 0（未标）、1（遮挡）或 2（可见）")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in (x, y)):
            raise ValueError("手部关键点需要有限的坐标")
        if (visible and not (left <= x <= right and top <= y <= bottom)) or (not visible and (x or y)):
            raise ValueError("已标关键点必须位于手框内，未标关键点的坐标必须为零")
    if not any(point[2] for point in keypoints):
        raise ValueError("每只手至少需要标一个关键点")


def read_mask(shape):
    points = shape["points"]
    if len(points) != 2 or any(type(value) is not int for point in points for value in point):
        raise ValueError("掩膜需要两个整数边界坐标")
    (left, top), (right, bottom) = points
    width, height = right - left, bottom - top
    encoded = shape.get("mask")
    if min(width, height) < 1 or width * height > MAX_MASK_PIXELS:
        raise ValueError("掩膜范围无效或像素过多")
    if not isinstance(encoded, str) or len(encoded) > 10_000_000:
        raise ValueError("掩膜需要有效的 PNG 数据")
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (width, height):
                raise ValueError("掩膜必须是与边界对应的单通道 PNG")
            histogram = image.histogram()
            if not histogram[255] or sum(histogram[1:255]):
                raise ValueError("掩膜需要二值前景像素")
            return image.copy()
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("掩膜 PNG 无法解码") from exc


def mask_shape(image):
    bounds = image.getbbox()
    if not bounds:
        return None
    with io.BytesIO() as buffer:
        image.crop(bounds).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    left, top, right, bottom = bounds
    return {"shape_type": "mask", "mask": encoded, "points": [[left, top], [right, bottom]]}


def semantic_mask(shapes, names, width, height):
    if width * height > MAX_MASK_PIXELS:
        raise ValueError("语义掩膜最多支持 3200 万像素")
    mask = Image.new("L", (int(width), int(height)))
    draw = ImageDraw.Draw(mask)
    indices = {label: index for index, label in enumerate(names, 1)}
    for shape in shapes:
        value = 0 if shape.get("erase") else indices[shape["label"]]
        points = [tuple(point) for point in shape["points"]]
        kind = shape["shape_type"]
        if kind == "polygon":
            draw.polygon(points, fill=value)
        elif kind == "brush":
            radius = shape["radius"]
            if len(points) > 1:
                draw.line(points, fill=value, width=2 * radius + 1, joint="curve")
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
        else:
            foreground = read_mask(shape)
            mask.paste(value, (*points[0], *points[1]), foreground)
    return mask
