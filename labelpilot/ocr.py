import io
import json
import math
import re
from pathlib import Path

from PIL import Image


READABILITY = ("readable", "partial", "unreadable")
VALUE_ROLES = ("process_value", "setpoint", "state", "clock", "unknown", "not_applicable")


def text_points(shape):
    points = shape["points"]
    if shape["shape_type"] == "rectangle":
        horizontal, vertical = zip(*points)
        left, right, top, bottom = min(horizontal), max(horizontal), min(vertical), max(vertical)
        return [[left, top], [right, top], [right, bottom], [left, bottom]]
    return points


def validate_ocr_shape(shape, *, reviewed=False):
    text = shape.get("text_raw")
    readability = shape.get("readability", "readable")
    role = shape.get("value_role", "unknown")
    if text is not None and (not isinstance(text, str) or len(text) > 4096):
        raise ValueError("OCR 原文必须为不超过 4096 字符的文本或留空")
    if readability not in READABILITY or role not in VALUE_ROLES:
        raise ValueError("OCR 可读性或数值角色无效")
    if text and any(ord(character) < 32 or character in "\x85\u2028\u2029" for character in text):
        raise ValueError("每个 OCR 区域仅标一行；原文不能包含换行、制表或控制字符")
    if readability == "readable" and text == "###":
        raise ValueError("### 是 PaddleOCR 忽略标记；请标为部分可读或不可读")
    if reviewed and readability == "readable" and not (text and text.strip()):
        raise ValueError("可读 OCR 区域必须填写原文；看不清请设置可读性，不要填零")
    if readability == "unreadable" and text:
        raise ValueError("不可读区域请留空原文，不能填写猜测读数")
    for field in ("unit", "field_name", "channel", "parent_id"):
        value = shape.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 160):
            raise ValueError("OCR 单位、字段、通道和关联对象必须为不超过 160 字符的文本")
    value = shape.get("value_decimal")
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value)):
        raise ValueError("OCR 数值必须是保留精度的十进制字符串或留空")
    if readability != "readable" and value is not None:
        raise ValueError("未完整可读区域不能填写确定数值")
    points = text_points(shape)
    if len(points) != 4:
        raise ValueError("OCR 文字区域使用矩形或四点多边形；请按文字边缘依次点击四个角")
    turns = []
    for index, current in enumerate(points):
        following, after = points[(index + 1) % 4], points[(index + 2) % 4]
        turns.append((following[0] - current[0]) * (after[1] - following[1])
                     - (following[1] - current[1]) * (after[0] - following[0]))
    if not (all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)):
        raise ValueError("OCR 四点框不能交叉、凹陷或退化；请沿文字边缘顺序标注")


def detection_entries(document):
    entries = []
    for shape in document["shapes"]:
        validate_ocr_shape(shape, reviewed=True)
        readable = shape.get("readability", "readable") == "readable"
        entries.append({"transcription": shape["text_raw"] if readable else "###", "points": text_points(shape)})
    return entries


def recognition_crop(image, points):
    import cv2
    import numpy as np

    corners = np.asarray(points, dtype=np.float32)
    center = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))
    corners = corners[order]
    corners = np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)
    width = max(1, math.ceil(max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))))
    height = max(1, math.ceil(max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))))
    if width < 2 or height < 2:
        raise ValueError("可读文字区域太小，无法生成识别裁剪；请检查标注范围")
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    cropped = cv2.warpPerspective(np.asarray(image), transform, (width, height),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    result = Image.fromarray(cropped)
    if height >= width * 1.5:
        result = result.transpose(Image.Transpose.ROTATE_90)
    return result


def write_ocr_annotation(archive, document, image_name, subset, split_config, records):
    suffix = f"/{subset}" if split_config else ""
    stem = Path(image_name).stem
    entries = detection_entries(document)
    metadata = {**document, "imagePath": ("../../" if split_config else "../") + image_name, "imageData": None}
    archive.writestr(f"annotations{suffix}/{stem}.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    records.setdefault("ocr_detection", []).append(image_name + "\t" + json.dumps(entries, ensure_ascii=False))
    recognition = records.setdefault("ocr_recognition", [])
    if not any(entry["transcription"] != "###" for entry in entries):
        return
    with Image.open(io.BytesIO(archive.read(image_name))) as original:
        original.load()
        with original.convert("RGB") as image:
            for index, entry in enumerate(entries):
                if entry["transcription"] == "###":
                    continue
                crop = recognition_crop(image, entry["points"])
                crop_name = f"recognition{suffix}/{stem}_{index + 1:04d}.png"
                with io.BytesIO() as buffer:
                    crop.save(buffer, format="PNG")
                    archive.writestr(crop_name, buffer.getvalue())
                with Image.open(io.BytesIO(archive.read(crop_name))) as decoded:
                    if decoded.mode != "RGB" or decoded.size != crop.size or decoded.tobytes() != crop.tobytes():
                        raise ValueError("OCR 识别裁剪回读不一致")
                recognition.append(crop_name + "\t" + entry["transcription"])


def verify_ocr_annotation(archive, document, image_name, subset, split_config, records):
    suffix = f"/{subset}" if split_config else ""
    decoded = json.loads(archive.read(f"annotations{suffix}/{Path(image_name).stem}.json"))
    expected = {**document, "imagePath": ("../../" if split_config else "../") + image_name, "imageData": None}
    if decoded != expected:
        raise ValueError("OCR 原文、属性或坐标回读不一致")
    recorded_image, encoded = records["ocr_detection"][-1].split("\t", 1)
    if recorded_image != image_name or json.loads(encoded) != detection_entries(document):
        raise ValueError("OCR 检测标签与原标注不一致")


def finish_ocr_export(archive, records, manifest):
    counts = {}
    for subset, documents in records.items():
        label = subset if subset != "unassigned" else "all"
        counts[label] = {"images": len(documents.get("ocr_detection", [])),
                         "recognition_crops": len(documents.get("ocr_recognition", []))}
        for directory, key in (("det", "ocr_detection"), ("rec", "ocr_recognition")):
            content = "".join(line + "\n" for line in documents.get(key, []))
            filename = f"{directory}/{label}.txt"
            archive.writestr(filename, content)
            if archive.read(filename).decode("utf-8") != content:
                raise ValueError("OCR 标签清单回读不一致")
    manifest["ocr"] = {"splits": counts, "ignore_transcription": "###",
                        "recognition_policy": "human_readable_only", "crop_format": "lossless_png"}
    archive.writestr("OCR_README.txt", "PaddleOCR 数据集\n以本目录为 data_dir；det/*.txt 用于文字检测，rec/*.txt 用于文字识别。\n"
                    "每张原图只分配一次集合，裁剪继承原图集合；建议按来源组划分，避免同源泄漏。\n"
                    "仅完整可读区域生成识别裁剪；部分/不可读区域在检测标签中标 ###，完整原文与属性保存在 annotations。\n"
                    "确认无文字的图片在检测清单中保留 []，不生成识别样本；这不表示读数为零。\n"
                    "四点区域作透视校正，竖长区域逆时针转正；请抽查文字方向。未执行 OCR 训练或识别质量验证。\n")
