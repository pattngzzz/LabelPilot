import copy
import hashlib
import io
import math
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, __version__ as pillow_version

from .store import Problem
from .tasks import is_rotated_box


MAX_COPIES = 3
PRESETS = {
    "gentle": {
        "name": "轻量通用（推荐）",
        "description": "亮度 0.85–1.15、对比度 / 饱和度 0.90–1.10；不改变几何位置，适合先做保守增强。",
        "brightness": (0.85, 1.15), "contrast": (0.90, 1.10), "color": (0.90, 1.10),
    },
    "lighting": {
        "name": "光照变化",
        "description": "亮度 0.65–1.35、对比度 / 饱和度 0.80–1.20；模拟偏暗、偏亮和色彩变化，不翻转图片。",
        "brightness": (0.65, 1.35), "contrast": (0.80, 1.20), "color": (0.80, 1.20),
    },
    "quality": {
        "name": "画质扰动",
        "description": "亮度 / 对比度 0.90–1.10、高斯模糊半径 0.3–0.8 像素、JPEG 质量 65–85；小目标、铭牌细字请谨慎使用。",
        "brightness": (0.90, 1.10), "contrast": (0.90, 1.10), "color": (1.0, 1.0),
        "blur": (0.3, 0.8), "jpeg_quality": (65, 85),
    },
    "horizontal": {
        "name": "水平翻转 + 轻量变化",
        "description": "水平镜像并同步变换框 / 多边形，叠加轻量亮度和色彩变化；仅适合左右无关类别，文字 / 方向有含义时不要选。",
        "brightness": (0.85, 1.15), "contrast": (0.90, 1.10), "color": (0.90, 1.10),
        "horizontal_flip": True,
    },
    "rotation": {
        "name": "图片旋转（90° / 180° / 270°）",
        "description": "逆时针旋转 90°、180° 或 270°，保留完整画面并同步框 / 多边形；90° / 270° 交换宽高。每张原图按种子确定角度顺序，生成 3 张时覆盖三个角度。仅适合方向无关场景。",
        "brightness": (1.0, 1.0), "contrast": (1.0, 1.0), "color": (1.0, 1.0),
        "rotation": True,
    },
}


OCR_PRESETS = {
    "ocr_gentle": {
        "name": "OCR 轻量增强（推荐）",
        "description": "亮度 / 对比度 0.95–1.05，不模糊、不镜像、不旋转、不裁切；PNG 输出，文字原文与顶点完全保留。",
        "brightness": (0.95, 1.05), "contrast": (0.95, 1.05), "color": (1.0, 1.0),
    },
    "ocr_lighting": {
        "name": "OCR 光照变化",
        "description": "亮度 0.85–1.15、对比度 0.90–1.10，模拟轻微明暗变化；不改变文字位置，PNG 输出。",
        "brightness": (0.85, 1.15), "contrast": (0.90, 1.10), "color": (1.0, 1.0),
    },
    "ocr_quality": {
        "name": "OCR 轻度画质扰动",
        "description": "轻微模糊（0.15–0.35 像素）及 JPEG 质量 92–98 的压缩模拟，再保存 PNG；细小文字请先抽查，不适合时使用轻量方案。",
        "brightness": (0.97, 1.03), "contrast": (0.97, 1.03), "color": (1.0, 1.0),
        "blur": (0.15, 0.35), "jpeg_quality": (92, 98),
    },
}
ALL_PRESETS = {**PRESETS, **OCR_PRESETS}


def task_presets(task_type):
    if task_type == "ocr":
        return OCR_PRESETS
    if task_type in ("detect", "segment"):
        return PRESETS
    raise Problem("当前导出增强仅适用于目标检测、实例分割和 OCR", 422)


def augmentation_presets(task_type="detect"):
    available = task_presets(task_type)
    presets = [{"id": name, "name": preset["name"], "description": preset["description"], "preset_count": 1}
               for name, preset in available.items()]
    description = ("OCR 轻量、光照变化、轻度画质扰动三种方案分别从原图生成，不连续叠加；不镜像、不旋转、不裁切，PNG 输出。"
                   "细小文字请抽查增强后是否仍然清晰，无法保证所有原图增强后均适合训练。" if task_type == "ocr" else
                   "轻量通用、光照变化、画质扰动、水平翻转、图片旋转五种方案分别从原图生成，不连续叠加。"
                   "含镜像、旋转和模糊：文字、方向有含义或特别小的目标请谨慎，不适合时改选单一方案。")
    presets.append({"id": "all", "name": "全部 OCR 方案（分别生成）" if task_type == "ocr" else "全部方案（分别生成）",
                    "preset_count": len(available), "description": description})
    return {"presets": presets, "max_copies": MAX_COPIES}


def plan_augmentation(options, task_type="detect"):
    settings = options.get("augmentation")
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) - {"preset", "copies", "seed"}:
        raise Problem("增强设置必须包含 preset，以及可选的 copies、seed", 422)
    available = task_presets(task_type)
    preset = settings.get("preset")
    if not isinstance(preset, str) or preset not in {*available, "all"}:
        raise Problem("请选择当前任务支持的数据增强方案；OCR 不支持镜像、直角旋转或检测增强方案", 422)
    copies = settings.get("copies", 1)
    if isinstance(copies, bool) or not isinstance(copies, int) or not 1 <= copies <= MAX_COPIES:
        raise Problem(f"每张训练图片、每种方案额外生成数量必须为 1 到 {MAX_COPIES} 的整数", 422)
    seed = settings.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise Problem("增强随机种子必须为 0 到 4294967295 的整数", 422)
    if options.get("split") is None:
        raise Problem("数据增强需先划分训练 / 验证 / 测试集；仅训练数据可设为 100 / 0 / 0", 422)
    if not options.get("include_images", True):
        raise Problem("数据增强必须勾选包含图片，否则增强标注无法配对", 422)
    if not options.get("reviewed_only", True):
        raise Problem("数据增强仅支持已人工确认的标注，请勾选只导出已人工确认", 422)
    presets = list(available) if preset == "all" else [preset]
    return {"preset": preset, "presets": presets, "copies": copies, "variants_per_image": copies * len(presets),
            "seed": seed, "scope": "train_only",
            "keep_originals": True, "algorithm_version": 1, "engine": f"Pillow {pillow_version}",
            "output": "PNG" if task_type == "ocr" else "JPEG", "alpha_background": "white"}


def rotation_angles(source_id, source_sha256, settings):
    material = f"labelpilot-rotation-v1:{settings['seed']}:{source_id}:{source_sha256}"
    angles = [90, 180, 270]
    random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big")).shuffle(angles)
    return angles


def validate_augmented_sample(source, augmented, contents, settings, provenance):
    def reject(reason):
        raise Problem(f"增强标注一致性校验失败：{reason}；已中止导出，不会丢弃对象或自动修框", 422)

    width, height = source["imageWidth"], source["imageHeight"]
    preset = ALL_PRESETS[settings["preset"]]
    horizontal = preset.get("horizontal_flip", False)
    operations = provenance.get("operations", {})
    angle = operations.get("rotation_degrees", 0)
    if preset.get("rotation"):
        variant = provenance.get("variant")
        if isinstance(variant, bool) or not isinstance(variant, int) or not 1 <= variant <= settings["copies"]:
            reject("旋转副本编号无效")
        expected_angle = rotation_angles(provenance.get("source_id"), provenance.get("source_sha256"), settings)[variant - 1]
        if isinstance(angle, bool) or angle != expected_angle:
            reject("旋转角度与种子及副本编号不一致")
    elif angle != 0:
        reject("非旋转方案出现旋转操作")
    if provenance.get("preset") != settings["preset"] or operations.get("horizontal_flip") is not horizontal:
        reject("增强操作与所选方案不一致")
    size = (height, width) if angle in (90, 270) else (width, height)
    if (augmented.get("imageWidth"), augmented.get("imageHeight")) != size:
        reject("图片尺寸与几何变换不一致")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            if image.size != size or image.getexif().get(274, 1) != 1:
                reject("增强图片尺寸或方向与标注不一致")
            if settings["output"] == "PNG" and image.format != "PNG":
                reject("OCR 增强图片必须为 PNG")
    except (OSError, ValueError) as error:
        raise Problem("增强图片无法完整解码，已中止导出", 422) from error
    if augmented.get("labelpilot", {}).get("image_sha256") != hashlib.sha256(contents).hexdigest():
        reject("增强图片指纹不匹配")
    shapes = augmented.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != len(source["shapes"]):
        reject("标注对象数量发生变化")
    for index, (original, actual) in enumerate(zip(source["shapes"], shapes), 1):
        if not isinstance(actual, dict):
            reject(f"第 {index} 个对象格式无效")
        expected = copy.deepcopy(original)
        if horizontal:
            expected["points"] = [[width - point[0], point[1]] for point in original["points"]]
        if angle == 90:
            expected["points"] = [[point[1], width - point[0]] for point in original["points"]]
        elif angle == 180:
            expected["points"] = [[width - point[0], height - point[1]] for point in original["points"]]
        elif angle == 270:
            expected["points"] = [[height - point[1], point[0]] for point in original["points"]]
        if actual != expected:
            reject(f"第 {index} 个对象的标签、坐标或属性不匹配")
        allow_outside = is_rotated_box(original)
        for point in actual["points"]:
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
                    or not allow_outside and (not 0 <= point[0] <= size[0] or not 0 <= point[1] <= size[1])):
                reject(f"第 {index} 个对象坐标无效或越界")
    return len(shapes)


def augmentation_image(path):
    with Image.open(path) as source:
        if source.mode in ("RGBA", "LA") or "transparency" in source.info:
            rgba = source.convert("RGBA")
            original = Image.new("RGB", source.size, "white")
            original.paste(rgba, mask=rgba.getchannel("A"))
        else:
            original = source.convert("RGB")
    return original


def augmented_variants(path, document, source_id, source_sha256, settings):
    preset = ALL_PRESETS[settings["preset"]]
    angles = rotation_angles(source_id, source_sha256, settings) if preset.get("rotation") else None
    original = path if isinstance(path, Image.Image) else augmentation_image(path)
    for index in range(1, settings["copies"] + 1):
        seed_material = f"labelpilot-augmentation-v1:{settings['seed']}:{source_id}:{source_sha256}:{settings['preset']}:{index}"
        generator = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest(), "big"))
        operations = {key: round(generator.uniform(*preset[key]), 6) for key in ("brightness", "contrast", "color")}
        operations.update(horizontal_flip=preset.get("horizontal_flip", False),
                          blur_radius=round(generator.uniform(*preset.get("blur", (0, 0))), 6),
                          jpeg_quality=generator.randint(*preset.get("jpeg_quality", (100, 100) if settings["output"] == "PNG" else (95, 95))))
        image = ImageEnhance.Brightness(original).enhance(operations["brightness"])
        image = ImageEnhance.Contrast(image).enhance(operations["contrast"])
        image = ImageEnhance.Color(image).enhance(operations["color"])
        augmented = copy.deepcopy(document)
        if operations["horizontal_flip"]:
            image = ImageOps.mirror(image)
            for shape in augmented["shapes"]:
                shape["points"] = [[image.width - point[0], point[1]] for point in shape["points"]]
        if angles:
            angle = angles[index - 1]
            operations["rotation_degrees"] = angle
            width, height = image.size
            if angle == 90:
                image = image.transpose(Image.Transpose.ROTATE_90)
                for shape in augmented["shapes"]:
                    shape["points"] = [[point[1], width - point[0]] for point in shape["points"]]
            elif angle == 180:
                image = image.transpose(Image.Transpose.ROTATE_180)
                for shape in augmented["shapes"]:
                    shape["points"] = [[width - point[0], height - point[1]] for point in shape["points"]]
            else:
                image = image.transpose(Image.Transpose.ROTATE_270)
                for shape in augmented["shapes"]:
                    shape["points"] = [[height - point[1], point[0]] for point in shape["points"]]
            augmented["imageWidth"], augmented["imageHeight"] = image.size
        if operations["blur_radius"]:
            image = image.filter(ImageFilter.GaussianBlur(operations["blur_radius"]))
        if settings["output"] == "PNG" and "jpeg_quality" in preset:
            with io.BytesIO() as compressed:
                image.save(compressed, format="JPEG", quality=operations["jpeg_quality"], subsampling=0)
                compressed.seek(0)
                with Image.open(compressed) as decoded:
                    image = decoded.convert("RGB")
        with io.BytesIO() as buffer:
            if settings["output"] == "PNG":
                image.save(buffer, format="PNG")
            else:
                image.save(buffer, format="JPEG", quality=operations["jpeg_quality"], subsampling=0)
            contents = buffer.getvalue()
        provenance = {"source_id": source_id, "source_sha256": source_sha256,
                      "preset": settings["preset"], "variant": index, "operations": operations,
                      "algorithm_version": settings["algorithm_version"], "engine": settings["engine"]}
        augmented["checked"] = False
        augmented["labelpilot"] = {"status": "draft", "human_reviewed": False,
                                   "image_sha256": hashlib.sha256(contents).hexdigest(), "augmentation": provenance}
        if isinstance(augmented.get("preannotation"), dict):
            augmented["preannotation"].update(human_reviewed=False, negative_sample_confirmed=False,
                                             status="augmentation_unreviewed")
        yield index, contents, augmented, provenance
