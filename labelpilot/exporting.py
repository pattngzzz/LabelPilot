import copy
import hashlib
import json
import math
import random
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from .cleaning import clean_selection
from .store import Problem, Project, digest, now, read_annotation_snapshot, source_group


SPLIT_NAMES = ("train", "val", "test")
SPLIT_UNITS = ("image", "source_group", "directory")
YOLO_POLYGON_ERROR = "YOLO 检测格式仅支持矩形。请选择 COCO 或 Labelme 导出多边形"


def split_group(row, unit):
    if unit == "source_group":
        return source_group(row.get("source_group"))
    if unit == "directory":
        return PurePosixPath(row["relative"]).parent.as_posix()
    return None


def assign_groups(rows, unit, active, targets, rng):
    groups = {}
    missing = 0
    for row in rows:
        group = split_group(row, unit)
        if group is None:
            missing += 1
        else:
            groups.setdefault(group, []).append(row["id"])
    if missing:
        raise Problem(f"有 {missing} 张图片没有来源组。可补齐任务清单，或选择其他划分方式", 422)
    if len(groups) < len(active):
        raise Problem(f"只有 {len(groups)} 个可用分组，无法填满 {len(active)} 个集合。请扩大导出范围或将部分比例设为 0", 422)

    ordered = sorted(groups)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: len(groups[group]), reverse=True)
    counts = dict.fromkeys(SPLIT_NAMES, 0)
    assignments = {}
    for index, group in enumerate(ordered):
        empty = [name for name in active if counts[name] == 0]
        # Reserve a whole group for each remaining empty subset.
        candidates = empty if len(ordered) - index == len(empty) else active
        subset = max(candidates, key=lambda name: targets[name] - counts[name])
        for item_id in groups[group]:
            assignments[item_id] = subset
        counts[subset] += len(groups[group])
    return assignments, counts


def plan_splits(rows, options):
    settings = options.get("split")
    if settings is None:
        return {row["id"]: "unassigned" for row in rows}, None, None
    if not isinstance(settings, dict) or set(settings) - {*SPLIT_NAMES, "seed", "unit"}:
        raise Problem("划分设置必须包含 train、val、test 百分比，可指定随机种子和划分方式", 422)
    unit = settings.get("unit", "image")
    if unit not in SPLIT_UNITS:
        raise Problem("划分方式必须是按图片、按来源组或按原图文件夹", 422)
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
    targets = {name: len(rows) * ratios[name] / 100 for name in SPLIT_NAMES}
    rng = random.Random(seed)
    if unit != "image":
        assignments, counts = assign_groups(rows, unit, active, targets, rng)
        return assignments, {**ratios, "seed": seed, "unit": unit, "method": "seeded_group_balance"}, counts

    if len(rows) < len(active):
        raise Problem("图片数量不足以让每个非零比例的集合至少包含 1 张，请增加图片或将部分比例设为 0", 422)
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
    rng.shuffle(shuffled)
    assignments = {}
    offset = 0
    for name in SPLIT_NAMES:
        for row in shuffled[offset:offset + counts[name]]:
            assignments[row["id"]] = name
        offset += counts[name]
    return assignments, {**ratios, "seed": seed, "unit": "image", "method": "seeded_shuffle"}, counts


@dataclass(frozen=True)
class ExportPlan:
    project: Project
    format_name: str
    batch: str
    reviewed_only: bool
    include_images: bool
    rows: list
    excluded: int
    assignments: dict
    split_config: Optional[dict]
    split_counts: Optional[dict]
    cleaning: Optional[list]


def prepare_export(store, project_id, options, *, include_documents=False, checkpoint=None, allow_empty_cleaning=False):
    project = store.project(project_id)
    format_name = options.get("format", "yolo")
    if format_name not in ("yolo", "labelme", "coco"):
        raise Problem("导出格式必须为 yolo、labelme 或 coco", 422)
    reviewed_only = options.get("reviewed_only", True)
    include_images = options.get("include_images", True)
    clean = options.get("clean", False)
    if any(type(value) is not bool for value in (reviewed_only, include_images, clean)):
        raise Problem("导出开关必须为布尔值", 422)
    if clean and not reviewed_only:
        raise Problem("清洗版数据集仅纳入已人工确认的图片", 422)
    batch = options.get("batch")
    if batch is None:
        batch = ""
    if not isinstance(batch, str):
        raise Problem("批次必须为文本", 422)
    listing = store.listing(project_id, batch=batch, limit=1000000,
                            include_documents=include_documents and not clean)
    rows = [item for item in listing["items"] if item["status"] == "reviewed" or
            (not reviewed_only and item["status"] in ("imported", "predicted", "draft"))]
    if not rows:
        raise Problem("所选范围没有符合条件的标注。默认仅导出已人工确认的图片", 422)
    if len(rows) > 10000:
        raise Problem("请按批次导出，每次不超过 10000 张", 422)
    cleaning = None
    if clean:
        rows, cleaning = clean_selection(store, project, rows, include_documents=include_documents, checkpoint=checkpoint)
        if not rows and not allow_empty_cleaning:
            counts = cleaning_summary(cleaning)["counts"]
            raise Problem(f"清洗后没有可用图片。图片不可用 {counts['unreadable']} 张，标注异常 {counts['annotation']} 张，冲突副本 {counts['conflict']} 张", 422)
    assignments, split_config, split_counts = plan_splits(rows, options) if rows else ({}, None, None)
    return ExportPlan(project, format_name, batch, reviewed_only, include_images, rows,
                      listing["filtered"] - len(rows), assignments, split_config, split_counts, cleaning)


def cleaning_summary(excluded):
    counts = dict.fromkeys(("duplicate", "conflict", "unreadable", "annotation"), 0)
    examples = []
    for item in excluded:
        reason = item["reason"]
        counts[reason] += 1
        if counts[reason] <= 4:
            examples.append(item)
    return {"counts": counts, "examples": examples}


def export_fingerprint(plan, sources):
    item_versions = [
        (row["id"], row["photo_id"], row.get("version", 0), row["status"], row["source_group"],
         row["image_sha256"], plan.assignments[row["id"]], sources.get(row["id"])) for row in plan.rows
    ]
    snapshot = (plan.project.id, plan.project.names, plan.format_name, plan.batch, plan.reviewed_only,
                plan.include_images, plan.excluded, plan.split_config, plan.cleaning, item_versions)
    return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_export_document(store, project, document, width, height, status):
    metadata = document.get("labelpilot")
    confirmed_empty = isinstance(metadata, dict) and metadata.get("negative_sample_confirmed", False)
    store.validate(project, document, width, height, status, confirmed_empty)


def preview_export(store, project_id, options):
    with store.lock:
        plan = prepare_export(store, project_id, options, include_documents=True, allow_empty_cleaning=True)
        return describe_export(store, plan)


def describe_export(store, plan):
    subsets = [name for name in SPLIT_NAMES if plan.split_counts[name]] if plan.split_counts else ["unassigned"]
    class_counts = {name: dict.fromkeys(subsets, 0) for name in plan.project.names}
    source_splits = {}
    fingerprints = {}
    objects = confirmed_empty = unreviewed = unreviewed_empty = polygons = missing_sources = 0
    for row in plan.rows:
        item_id = row["id"]
        subset = plan.assignments[item_id]
        document = row.get("document")
        if document is None:
            try:
                source = plan.project.item(item_id)["source"]
                document, fingerprints[item_id] = read_annotation_snapshot(source)
            except (OSError, ValueError, TypeError) as exc:
                raise Problem(f"{row['photo_id']} 的原标注无法读取，请重新扫描数据", 409) from exc
        if not isinstance(document, dict):
            raise Problem(f"{row['photo_id']} 的标注格式无效", 422)
        width, height = document.get("imageWidth"), document.get("imageHeight")
        if any(type(value) not in (int, float) or not 0 < value or
               (type(value) is float and not value.is_integer()) for value in (width, height)):
            raise Problem(f"{row['photo_id']} 的标注缺少有效的原图尺寸", 422)
        try:
            validate_export_document(store, plan.project, document, width, height, row["status"])
        except Problem as exc:
            raise Problem(f"{row['photo_id']}：{exc.message}", exc.status) from exc
        shapes = document["shapes"]
        objects += len(shapes)
        if row["status"] == "reviewed":
            confirmed_empty += not shapes
        else:
            unreviewed += 1
            unreviewed_empty += not shapes
        polygons += any(shape["shape_type"] == "polygon" for shape in shapes)
        for shape in shapes:
            class_counts[shape["label"]][subset] += 1
        group = source_group(row["source_group"])
        if group is None:
            missing_sources += 1
        else:
            source_splits.setdefault(group, set()).add(subset)

    blocking_issue = None
    if not plan.rows:
        blocking_issue = "清洗后没有可用图片。请检查排除示例或扩大批次范围"
    elif plan.format_name == "yolo" and polygons:
        blocking_issue = f"{polygons} 张图片包含多边形。{YOLO_POLYGON_ERROR}"
    return {
        "images": len(plan.rows), "excluded": plan.excluded, "annotations": objects,
        "confirmed_empty": confirmed_empty, "unreviewed": unreviewed, "unreviewed_empty": unreviewed_empty,
        "split_counts": plan.split_counts, "class_counts": [
            {"name": name, "total": sum(counts.values()), "splits": counts} for name, counts in class_counts.items()
        ],
        "missing_sources": missing_sources, "cross_split_sources": sum(len(splits) > 1 for splits in source_splits.values()),
        "group_count": len({split_group(row, plan.split_config["unit"]) for row in plan.rows})
        if plan.split_config and plan.split_config["unit"] != "image" else None,
        "blocking_issue": blocking_issue,
        "cleaning": cleaning_summary(plan.cleaning) if plan.cleaning is not None else None,
        "preview_token": export_fingerprint(plan, fingerprints),
    }


def export_project(store, project_id, options):
    with store.lock:
        plan = prepare_export(store, project_id, options)
        fingerprints = {}
        expected = options.get("preview_token")
        if expected is not None:
            for row in plan.rows:
                if "version" not in row:
                    try:
                        fingerprints[row["id"]] = digest(plan.project.item(row["id"])["source"])
                    except OSError as exc:
                        raise Problem(f"{row['photo_id']} 的原标注无法读取，请重新扫描数据", 409) from exc
            if expected != export_fingerprint(plan, fingerprints):
                raise Problem("数据已变化，请查看更新后的预检结果再导出", 409)
        return write_archive(store, plan, fingerprints)


def write_archive(store, plan, fingerprints):
    project_id, format_name = plan.project.id, plan.format_name
    folder = store.home / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    export_id = uuid.uuid4().hex
    path = folder / f"{project_id}_{format_name}_{export_id}.zip"
    temporary = path.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
            write_members(store, plan, archive, fingerprints)
        temporary.replace(path)
        return {"id": path.stem, "filename": path.name, "images": len(plan.rows), "split_counts": plan.split_counts,
                "url": f"/api/exports/{path.stem}"}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class DatasetFolder:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True)

    def open(self, name, mode):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode + "b")

    def writestr(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_verified_image(archive, source, name, expected_digest):
    hasher = hashlib.sha256()
    with Path(source).open("rb") as original, archive.open(name, "w") as destination:
        for block in iter(lambda: original.read(1024 * 1024), b""):
            hasher.update(block)
            destination.write(block)
    if hasher.hexdigest() != expected_digest:
        raise Problem("复制图片时原图发生变化，请重新准备数据", 409)


def write_members(store, plan, archive, fingerprints, *, checkpoint=None):
    project, format_name, rows = plan.project, plan.format_name, plan.rows
    split_config, split_counts = plan.split_config, plan.split_counts
    unit = split_config["unit"] if split_config else None
    project_id = project.id
    manifest = {"project": project_id, "format": format_name, "created_at": now(),
                "reviewed_only": plan.reviewed_only, "include_images": plan.include_images,
                "classes": project.names, "split": {"image": "random_by_image", "source_group": "grouped_by_source",
                                                     "directory": "grouped_by_directory"}.get(unit, "unassigned"),
                "split_config": split_config, "split_counts": split_counts,
                "source_writes": 0, "items": [], "warning": "评估请使用人工审核的样本。" +
                {"image": "按图片随机划分，同源 / 视频连续帧及重复图片可能跨集",
                 "source_group": "按任务清单来源组划分，实际比例取决于各组图量",
                 "directory": "按原图文件夹划分，分散在不同文件夹的同源图片可能跨集"}.get(unit, "未划分 train/val/test")}
    subsets = [name for name in SPLIT_NAMES if split_counts[name]] if split_config else ["unassigned"]
    coco_documents = {name: {"images": [], "annotations": [], "categories": [
        {"id": index + 1, "name": label} for index, label in enumerate(project.names)]} for name in subsets}
    if plan.cleaning is not None:
        manifest["cleaning"] = plan.cleaning
    for image_index, row in enumerate(rows, 1):
        if checkpoint:
            checkpoint()
        item_id = row["id"]
        current = store.detail(project_id, item_id)
        if current["version"] != row.get("version", 0) or current["status"] != row["status"]:
            raise Problem("导出期间标注发生变化，请重试", 409)
        if item_id in fingerprints and current["source_fingerprint"] != fingerprints[item_id]:
            raise Problem("导出期间原标注发生变化，请重试", 409)
        if plan.cleaning is not None and current["image_sha256"] != row["image_sha256"]:
            raise Problem("图像在清洗后发生变化，请重新预检", 409)
        document = copy.deepcopy(current["document"])
        validate_export_document(store, project, document, current["width"], current["height"], current["status"])
        item = project.item(row["id"])
        safe_photo_id = re.sub(r"[^\w.-]", "_", row["photo_id"])
        filename = f"{safe_photo_id}_{row['id'][:8]}{Path(item['path']).suffix.lower()}"
        subset = plan.assignments[item_id]
        suffix = f"/{subset}" if split_config else ""
        image_name = f"images{suffix}/{filename}"
        base_name = Path(filename).stem
        manifest["items"].append({"photo_id": row["photo_id"], "id": item_id, "status": current["status"],
                                   "revision": current["version"], "sha256": current["image_sha256"],
                                   "source_group": item["source_group"], "image": image_name, "split": subset})
        group = split_group(row, unit)
        if group is not None:
            manifest["items"][-1]["split_group"] = group
        if plan.include_images:
            if plan.cleaning is not None:
                write_verified_image(archive, item["path"], image_name, current["image_sha256"])
            else:
                archive.write(item["path"], image_name)
        if format_name == "labelme":
            document["imagePath"] = ("../../" if split_config else "../") + image_name
            document["imageData"] = None
            for shape in document["shapes"]:
                if shape["shape_type"] == "rectangle":
                    xs, ys = zip(*shape["points"])
                    shape["points"] = [[min(xs), min(ys)], [max(xs), max(ys)]]
            archive.writestr(f"annotations{suffix}/{base_name}.json", json.dumps(document, ensure_ascii=False, indent=2))
        elif format_name == "yolo":
            lines = []
            for shape in document["shapes"]:
                if shape["shape_type"] != "rectangle":
                    raise Problem(YOLO_POLYGON_ERROR, 422)
                xs, ys = zip(*shape["points"])
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                values = [(left + right) / 2 / current["width"], (top + bottom) / 2 / current["height"],
                          (right - left) / current["width"], (bottom - top) / current["height"]]
                lines.append(str(project.names.index(shape["label"])) + " " + " ".join(f"{value:.12f}" for value in values))
            archive.writestr(f"labels{suffix}/{base_name}.txt", "\n".join(lines) + ("\n" if lines else ""))
        else:
            coco = coco_documents[subset]
            coco["images"].append({"id": image_index, "file_name": image_name, "width": current["width"], "height": current["height"]})
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
    archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
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
    return manifest
