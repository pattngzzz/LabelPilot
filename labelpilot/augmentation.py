import copy
import hashlib
import io
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, __version__ as pillow_version

from .store import Problem


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
}


def augmentation_presets():
    return {"presets": [{"id": name, "name": preset["name"], "description": preset["description"]}
                        for name, preset in PRESETS.items()], "max_copies": MAX_COPIES}


def plan_augmentation(options):
    settings = options.get("augmentation")
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) - {"preset", "copies", "seed"}:
        raise Problem("增强设置必须包含 preset，以及可选的 copies、seed", 422)
    preset = settings.get("preset")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise Problem("请选择有效的数据增强方案", 422)
    copies = settings.get("copies", 1)
    if isinstance(copies, bool) or not isinstance(copies, int) or not 1 <= copies <= MAX_COPIES:
        raise Problem(f"每张训练图片额外生成数量必须为 1 到 {MAX_COPIES} 的整数", 422)
    seed = settings.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise Problem("增强随机种子必须为 0 到 4294967295 的整数", 422)
    if options.get("split") is None:
        raise Problem("数据增强需先划分训练 / 验证 / 测试集；仅训练数据可设为 100 / 0 / 0", 422)
    if not options.get("include_images", True):
        raise Problem("数据增强必须勾选包含图片，否则增强标注无法配对", 422)
    if not options.get("reviewed_only", True):
        raise Problem("数据增强仅支持已人工确认的标注，请勾选只导出已人工确认", 422)
    return {"preset": preset, "copies": copies, "seed": seed, "scope": "train_only",
            "keep_originals": True, "algorithm_version": 1, "engine": f"Pillow {pillow_version}",
            "output": "JPEG", "alpha_background": "white"}


def augmented_variants(path, document, source_id, source_sha256, settings):
    preset = PRESETS[settings["preset"]]
    with Image.open(path) as source:
        if source.mode in ("RGBA", "LA") or "transparency" in source.info:
            rgba = source.convert("RGBA")
            original = Image.new("RGB", source.size, "white")
            original.paste(rgba, mask=rgba.getchannel("A"))
        else:
            original = source.convert("RGB")
    for index in range(1, settings["copies"] + 1):
        seed_material = f"labelpilot-augmentation-v1:{settings['seed']}:{source_id}:{source_sha256}:{settings['preset']}:{index}"
        generator = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest(), "big"))
        operations = {key: round(generator.uniform(*preset[key]), 6) for key in ("brightness", "contrast", "color")}
        operations.update(horizontal_flip=preset.get("horizontal_flip", False),
                          blur_radius=round(generator.uniform(*preset.get("blur", (0, 0))), 6),
                          jpeg_quality=generator.randint(*preset.get("jpeg_quality", (95, 95))))
        image = ImageEnhance.Brightness(original).enhance(operations["brightness"])
        image = ImageEnhance.Contrast(image).enhance(operations["contrast"])
        image = ImageEnhance.Color(image).enhance(operations["color"])
        augmented = copy.deepcopy(document)
        if operations["horizontal_flip"]:
            image = ImageOps.mirror(image)
            for shape in augmented["shapes"]:
                shape["points"] = [[image.width - point[0], point[1]] for point in shape["points"]]
        if operations["blur_radius"]:
            image = image.filter(ImageFilter.GaussianBlur(operations["blur_radius"]))
        with io.BytesIO() as buffer:
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
