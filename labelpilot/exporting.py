import copy
import hashlib
import io
import json
import math
import random
import re
import uuid
import zipfile
from pathlib import Path

from .augmentation import augmented_variants, plan_augmentation, validate_augmented_sample
from .store import Problem, now


SPLIT_NAMES = ("train", "val", "test")


def plan_splits(rows, options):
    settings = options.get("split")
    if settings is None:
        return {row["id"]: "unassigned" for row in rows}, None, None
    if not isinstance(settings, dict) or set(settings) - {*SPLIT_NAMES, "seed"}:
        raise Problem("划分设置必须包含 train、val、test 百分比和可选随机种子", 422)
    ratios = {name: settings.get(name) for name in SPLIT_NAMES}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not 0 <= value <= 100 for value in ratios.values()):
        raise Problem("训练 / 验证 / 测试比例必须为 0 到 100 的数字", 422)
    if not math.isclose(math.fsum(ratios.values()), 100, rel_tol=0, abs_tol=1e-6) or ratios["train"] <= 0:
        raise Problem("三个比例之和必须为 100%，且训练集比例必须大于 0", 422)
    seed = settings.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise Problem("随机种子必须为 0 到 4294967295 的整数", 422)
    active = [name for name in SPLIT_NAMES if ratios[name] > 0]
    if len(rows) < len(active):
        raise Problem("图片数量不足以让每个非零比例的集合至少包含 1 张，请增加图片或将部分比例设为 0", 422)
    targets = {name: len(rows) * ratios[name] / 100 for name in SPLIT_NAMES}
    counts = {name: math.floor(targets[name]) for name in SPLIT_NAMES}
    order = sorted(active, key=lambda name: targets[name] - counts[name], reverse=True)
    for name in order[:len(rows) - sum(counts.values())]:
        counts[name] += 1
    for name in active:
        if not counts[name]:
            donor = max((candidate for candidate in active if counts[candidate] > 1),
                        key=lambda candidate: counts[candidate] - targets[candidate])
            counts[donor] -= 1
            counts[name] = 1
    shuffled = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(shuffled)
    assignments = {}
    offset = 0
    for name in SPLIT_NAMES:
        for row in shuffled[offset:offset + counts[name]]:
            assignments[row["id"]] = name
        offset += counts[name]
    return assignments, {**ratios, "seed": seed, "unit": "image", "method": "seeded_shuffle"}, counts


def write_annotation(archive, format_name, project, document, image_name, subset, split_config, coco_documents, image_index):
    suffix = f"/{subset}" if split_config else ""
    base_name = Path(image_name).stem
    if format_name == "labelme":
        document["imagePath"] = ("../../" if split_config else "../") + image_name
        document["imageData"] = None
        archive.writestr(f"annotations{suffix}/{base_name}.json", json.dumps(document, ensure_ascii=False, indent=2))
    elif format_name == "yolo":
        lines = []
        for shape in document["shapes"]:
            if shape["shape_type"] != "rectangle":
                raise Problem("YOLO 检测导出不支持多边形；请改选 COCO / Labelme，避免静默转换", 422)
            xs, ys = zip(*shape["points"])
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            values = [(left + right) / 2 / document["imageWidth"], (top + bottom) / 2 / document["imageHeight"],
                      (right - left) / document["imageWidth"], (bottom - top) / document["imageHeight"]]
            lines.append(str(project.names.index(shape["label"])) + " " + " ".join(f"{value:.12f}" for value in values))
        archive.writestr(f"labels{suffix}/{base_name}.txt", "\n".join(lines) + ("\n" if lines else ""))
    else:
        coco = coco_documents[subset]
        coco["images"].append({"id": image_index, "file_name": image_name,
                               "width": document["imageWidth"], "height": document["imageHeight"]})
        for shape in document["shapes"]:
            xs, ys = zip(*shape["points"])
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            polygon = shape["points"]
            if shape["shape_type"] == "polygon":
                area = abs(sum(polygon[index][0] * polygon[(index + 1) % len(polygon)][1] -
                               polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                               for index in range(len(polygon)))) / 2
            else:
                area = width * height
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_index,
                                        "category_id": project.names.index(shape["label"]) + 1,
                                        "bbox": [min(xs), min(ys), width, height], "area": area, "iscrowd": 0,
                                        "segmentation": [[value for point in polygon for value in point]] if shape["shape_type"] == "polygon" else []})


def verify_export_annotation(archive, format_name, project, document, image_name, subset, split_config,
                             coco_documents, image_index, annotation_offset):
    def reject(reason):
        raise Problem(f"导出标注回读校验失败：{reason}；已中止导出", 422)

    suffix = f"/{subset}" if split_config else ""
    base_name = Path(image_name).stem
    shapes = document["shapes"]
    if format_name == "labelme":
        decoded = json.loads(archive.read(f"annotations{suffix}/{base_name}.json"))
        if (decoded.get("shapes") != shapes or decoded.get("imageWidth") != document["imageWidth"]
                or decoded.get("imageHeight") != document["imageHeight"]
                or decoded.get("imagePath") != ("../../" if split_config else "../") + image_name):
            reject("Labelme 对象、尺寸或图片配对不一致")
    elif format_name == "yolo":
        lines = archive.read(f"labels{suffix}/{base_name}.txt").decode().splitlines()
        if len(lines) != len(shapes):
            reject("YOLO 框数量不一致")
        width, height = document["imageWidth"], document["imageHeight"]
        tolerance = max(width, height) * 1e-12 + 1e-9
        for line, shape in zip(lines, shapes):
            values = line.split()
            try:
                if len(values) != 5 or int(values[0]) != project.names.index(shape["label"]):
                    reject("YOLO 类别或字段不一致")
                center_x, center_y, box_width, box_height = map(float, values[1:])
            except ValueError as error:
                raise Problem("YOLO 导出坐标无法解析，已中止导出", 422) from error
            actual = [(center_x - box_width / 2) * width, (center_y - box_height / 2) * height,
                      (center_x + box_width / 2) * width, (center_y + box_height / 2) * height]
            xs, ys = zip(*shape["points"])
            expected = [min(xs), min(ys), max(xs), max(ys)]
            if (box_width <= 0 or box_height <= 0 or any(not math.isfinite(value) for value in actual)
                    or any(abs(first - second) > tolerance for first, second in zip(actual, expected))):
                reject("YOLO 框坐标与原标注不一致")
    else:
        coco = coco_documents[subset]
        expected_image = {"id": image_index, "file_name": image_name,
                          "width": document["imageWidth"], "height": document["imageHeight"]}
        if not coco["images"] or coco["images"][-1] != expected_image:
            reject("COCO 图片配对或尺寸不一致")
        annotations = coco["annotations"][annotation_offset:]
        if len(annotations) != len(shapes):
            reject("COCO 对象数量不一致")
        for actual, shape in zip(annotations, shapes):
            xs, ys = zip(*shape["points"])
            expected_box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            expected_polygon = [[coordinate for point in shape["points"] for coordinate in point]] if shape["shape_type"] == "polygon" else []
            if (actual.get("image_id") != image_index or actual.get("category_id") != project.names.index(shape["label"]) + 1
                    or actual.get("bbox") != expected_box or actual.get("segmentation") != expected_polygon):
                reject("COCO 类别、坐标或多边形不一致")


def export_project(store, project_id, options):
    project = store.project(project_id)
    format_name = options.get("format", "yolo")
    if format_name not in ("yolo", "labelme", "coco"):
        raise Problem("导出格式必须为 yolo、labelme 或 coco", 422)
    reviewed_only = options.get("reviewed_only", True)
    if not isinstance(reviewed_only, bool) or not isinstance(options.get("include_images", True), bool):
        raise Problem("导出开关必须为布尔值", 422)
    augmentation = plan_augmentation(options)
    listing = store.listing(project_id, batch=options.get("batch", ""), limit=1000000)
    rows = [item for item in listing["items"] if item["status"] == "reviewed" or
            (not reviewed_only and item["status"] in ("imported", "predicted", "draft"))]
    if not rows:
        raise Problem("没有符合条件的标注；默认只导出已由人工确认的图片", 422)
    if len(rows) > 10000:
        raise Problem("请按批次导出，每次不超过 10000 张", 422)
    assignments, split_config, split_counts = plan_splits(rows, options)
    source_split_counts = copy.deepcopy(split_counts)
    augmented_images = split_counts["train"] * augmentation["copies"] if augmentation else 0
    if len(rows) + augmented_images > 20000:
        raise Problem("增强后每次导出最多 20000 张，请减少副本数量或按批次导出", 422)
    if augmentation:
        split_counts["train"] += augmented_images
    folder = store.home / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    export_id = uuid.uuid4().hex
    path = folder / f"{project_id}_{format_name}_{export_id}.zip"
    manifest = {"project": project_id, "format": format_name, "created_at": now(),
                "reviewed_only": reviewed_only, "include_images": options.get("include_images", True),
                "classes": project.names, "split": "random_by_image" if split_config else "unassigned",
                "split_config": split_config, "split_counts": split_counts, "source_split_counts": source_split_counts,
                "original_images": len(rows), "augmented_images": augmented_images, "augmentation": augmentation,
                "source_writes": 0, "items": [], "warning": "模型预标注不等于真值；" +
                ("按图片随机划分，同源 / 视频连续帧及重复图片可能跨集，需人工检查泄漏" if split_config else "未划分 train/val/test")}
    subsets = [name for name in SPLIT_NAMES if split_counts[name]] if split_config else ["unassigned"]
    coco_documents = {name: {"images": [], "annotations": [], "categories": [
        {"id": index + 1, "name": label} for index, label in enumerate(project.names)]} for name in subsets}
    try:
        with store.lock, zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
            for row in rows:
                current = store.detail(project_id, row["id"])
                if reviewed_only and current["status"] != "reviewed":
                    raise Problem("导出期间审核状态发生变化，请重试", 409)
                document = copy.deepcopy(current["document"])
                source_document = copy.deepcopy(document) if augmentation else None
                store.validate(project, document, current["width"], current["height"], current["status"],
                               document.get("labelpilot", {}).get("negative_sample_confirmed", False))
                item = project.item(row["id"])
                safe_photo_id = re.sub(r"[^\w.-]", "_", row["photo_id"])
                filename = f"{safe_photo_id}_{row['id'][:8]}{Path(item['path']).suffix.lower()}"
                subset = assignments[row["id"]]
                suffix = f"/{subset}" if split_config else ""
                image_name = f"images{suffix}/{filename}"
                manifest["items"].append({"photo_id": row["photo_id"], "id": row["id"], "status": current["status"],
                                           "revision": current["version"], "sha256": current["image_sha256"],
                                           "source_group": item["source_group"], "image": image_name, "split": subset})
                if options.get("include_images", True):
                    if augmentation:
                        source_contents = Path(item["path"]).read_bytes()
                        archive.writestr(image_name, source_contents)
                    else:
                        archive.write(item["path"], image_name)
                annotation_offset = len(coco_documents[subset]["annotations"])
                write_annotation(archive, format_name, project, document, image_name, subset, split_config,
                                 coco_documents, len(manifest["items"]))
                if augmentation:
                    if hashlib.sha256(archive.read(image_name)).hexdigest() != current["image_sha256"]:
                        raise Problem("原图副本指纹与已保存标注不一致，已中止导出", 409)
                    verify_export_annotation(archive, format_name, project, source_document, image_name, subset,
                                             split_config, coco_documents, len(manifest["items"]), annotation_offset)
                    manifest["items"][-1]["annotation_count"] = len(source_document["shapes"])
                if augmentation and subset == "train":
                    produced = 0
                    for index, contents, augmented, provenance in augmented_variants(
                            io.BytesIO(source_contents), copy.deepcopy(source_document), row["id"], current["image_sha256"], augmentation):
                        if (index != produced + 1 or index > augmentation["copies"] or provenance.get("variant") != index
                                or provenance.get("source_id") != row["id"] or provenance.get("source_sha256") != current["image_sha256"]):
                            raise Problem("增强副本编号或来源不一致，已中止导出", 422)
                        produced += 1
                        shape_count = validate_augmented_sample(source_document, augmented, contents, augmentation, provenance)
                        variant_id = f"{row['id']}__aug_{augmentation['preset']}_{index}"
                        variant_name = f"images/train/{Path(filename).stem}__aug_{augmentation['preset']}_{index}.jpg"
                        archive.writestr(variant_name, contents)
                        if hashlib.sha256(archive.read(variant_name)).hexdigest() != augmented["labelpilot"]["image_sha256"]:
                            raise Problem("增强图片 ZIP 回读指纹不匹配，已中止导出", 422)
                        manifest["items"].append({"photo_id": row["photo_id"], "id": variant_id, "status": "augmented",
                                                  "revision": 0, "sha256": augmented["labelpilot"]["image_sha256"],
                                                  "source_group": item["source_group"], "image": variant_name,
                                                  "split": "train", "source_id": row["id"],
                                                  "source_revision": current["version"], "human_reviewed": False,
                                                  "annotation_count": shape_count,
                                                  "augmentation": provenance})
                        expected_document = copy.deepcopy(augmented)
                        annotation_offset = len(coco_documents["train"]["annotations"])
                        write_annotation(archive, format_name, project, augmented, variant_name, "train", split_config,
                                         coco_documents, len(manifest["items"]))
                        verify_export_annotation(archive, format_name, project, expected_document, variant_name, "train",
                                                 split_config, coco_documents, len(manifest["items"]), annotation_offset)
                    if produced != augmentation["copies"]:
                        raise Problem("增强副本数量缺失，已中止导出", 422)
            archive.writestr("classes.txt", "\n".join(project.names) + "\n")
            if format_name == "yolo":
                paths = "" if not split_config else "".join(
                    f"{name}: " + (f"images/{name}" if split_counts[name] else "null") + "\n" for name in SPLIT_NAMES)
                archive.writestr("dataset.yaml", paths + "names:\n" + "\n".join(
                    f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(project.names)) + "\n")
            if format_name == "coco":
                for name, document in coco_documents.items():
                    filename = f"annotations/{name}.json" if split_config else "annotations.json"
                    archive.writestr(filename, json.dumps(document, ensure_ascii=False))
                    if augmentation and json.loads(archive.read(filename)) != document:
                        raise Problem("COCO ZIP 回读与已校验标注不一致，已中止导出", 422)
            if augmentation:
                if len(manifest["items"]) != len(rows) + augmented_images:
                    raise Problem("导出图片总数与计划不一致，已中止导出", 422)
                if len(archive.namelist()) != len(set(archive.namelist())):
                    raise Problem("导出文件名冲突，已中止导出，避免图片或标注被覆盖", 422)
                manifest["integrity_check"] = {"passed": True, "images_checked": len(manifest["items"]),
                                               "augmented_images_checked": augmented_images,
                                               "annotations_checked": sum(entry["annotation_count"] for entry in manifest["items"]),
                                               "checks": ["image_sha256", "decoded_augmented_dimensions", "object_count",
                                                          "class_and_attributes", "exact_transformed_points", "serialized_annotation_readback",
                                                          "unique_archive_paths"]}
            archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return {"id": path.stem, "filename": path.name, "images": len(manifest["items"]), "split_counts": split_counts,
                "original_images": len(rows), "augmented_images": augmented_images, "source_split_counts": source_split_counts,
                "integrity_check": manifest.get("integrity_check"),
                "url": f"/api/exports/{path.stem}"}
    except Exception:
        path.unlink(missing_ok=True)
        raise
