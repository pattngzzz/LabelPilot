import math

from .store import Problem


def duplicate_threshold(options):
    value = options.get("duplicate_iou", 0.65)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.5 <= value <= 1:
        raise Problem("重复框 IoU 必须为 0.5 到 1 之间的数字", 422)
    return float(value)


def bounds(shape):
    points = shape["points"]
    return min(point[0] for point in points), min(point[1] for point in points), max(point[0] for point in points), max(point[1] for point in points)


def box_iou(first, second):
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    intersection = max(0, min(right, other_right) - max(left, other_left)) * max(0, min(bottom, other_bottom) - max(top, other_top))
    union = (right - left) * (bottom - top) + (other_right - other_left) * (other_bottom - other_top) - intersection
    return intersection / union if union > 0 else 0


def filter_duplicates(shapes, threshold, *, protected=()):
    def score(index):
        value = shapes[index].get("score", 0)
        return value if type(value) in (int, float) and math.isfinite(value) else 0

    kept = list(protected)
    removed = set()
    for index in sorted(range(len(shapes)), key=lambda index: (-score(index), index)):
        shape = shapes[index]
        if shape.get("shape_type") == "rectangle":
            rectangle = bounds(shape)
            if any(other.get("shape_type") == "rectangle" and other.get("label") == shape.get("label")
                   and box_iou(rectangle, bounds(other)) >= threshold for other in kept):
                removed.add(index)
                continue
        kept.append(shape)
    return [shape for index, shape in enumerate(shapes) if index not in removed], sorted(removed)
