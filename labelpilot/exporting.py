import copy
import hashlib
import io
import json
import math
import random
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from PIL import Image

from .augmentation import augmentation_image, augmented_variants, plan_augmentation, validate_augmented_sample
from .export_acceleration import cache_key
from .cleaning import clean_selection
from .store import Problem, Project, digest, now, read_annotation_snapshot, source_group
from .tasks import EXPORT_FORMATS, HAND_COUNT, HAND_KEYPOINTS, HAND_SKELETON, model_labels, polygon_area, semantic_mask


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
    augmentation: Optional[dict] = None
    output_task: Optional[str] = None

    @property
    def augmented_images(self):
        return self.split_counts["train"] * self.augmentation["variants_per_image"] if self.split_counts and self.augmentation else 0


def output_document(plan, document):
    document = copy.deepcopy(document)
    if plan.output_task == "detect":
        for shape in document["shapes"]:
            xs, ys = zip(*shape["points"])
            shape.update(shape_type="rectangle", points=[[min(xs), min(ys)], [max(xs), max(ys)]])
            shape.pop("contour_source", None)
    elif plan.output_task == "segment" or plan.project.task_type == "segment" and plan.format_name == "yolo":
        if any(shape["shape_type"] != "polygon" for shape in document["shapes"]):
            raise Problem("实例分割导出仍有未生成轮廓的检测框；请补全轮廓后导出，不会静默丢弃对象", 422)
    return document


def output_project(plan):
    project = copy.copy(plan.project)
    if plan.output_task:
        project.task_type = plan.output_task
    return project


def prepare_export(store, project_id, options, *, include_documents=False, checkpoint=None, allow_empty_cleaning=False):
    project = store.project(project_id)
    output_task = options.get("output_task")
    if output_task is not None and (project.task_type not in ("detect", "segment") or output_task not in ("detect", "segment")):
        raise Problem("检测 / 实例分割导出仅支持框和轮廓项目", 422)
    formats = EXPORT_FORMATS[project.task_type]
    format_name = options.get("format", formats[0])
    if format_name not in formats:
        raise Problem("请选择当前任务支持的导出格式", 422)
    reviewed_only = options.get("reviewed_only", True)
    include_images = options.get("include_images", True)
    clean = options.get("clean", False)
    if any(type(value) is not bool for value in (reviewed_only, include_images, clean)):
        raise Problem("导出开关必须为布尔值", 422)
    augmentation = plan_augmentation(options)
    if augmentation and project.task_type in ("semantic", "hand_pose"):
        raise Problem("当前导出增强仅适用于目标检测和实例分割", 422)
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
    plan = ExportPlan(project, format_name, batch, reviewed_only, include_images, rows,
                      listing["filtered"] - len(rows), assignments, split_config, split_counts, cleaning, augmentation, output_task)
    if len(rows) + plan.augmented_images > 20000:
        raise Problem("增强后每次导出最多 20000 张，请减少副本数量或按批次导出", 422)
    return plan


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
                plan.include_images, plan.excluded, plan.split_config, plan.cleaning, plan.augmentation, plan.output_task, item_versions)
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
    projection_issue = None
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
        document = plan.project.normalize_labels(document)
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
        try:
            shapes = output_document(plan, document)["shapes"]
        except Problem as error:
            projection_issue = error.message
        labels = [shape["label"] for shape in shapes]
        if plan.project.task_type == "semantic":
            mask = semantic_mask(shapes, plan.project.names, width, height)
            labels = [plan.project.names[value - 1] for _, value in mask.getcolors(256) if value]
        objects += len(labels)
        if row["status"] == "reviewed":
            confirmed_empty += not labels
        else:
            unreviewed += 1
            unreviewed_empty += not labels
        polygons += any(shape["shape_type"] == "polygon" for shape in shapes)
        for label in labels:
            class_counts[label][subset] += 1
        group = source_group(row["source_group"])
        if group is None:
            missing_sources += 1
        else:
            source_splits.setdefault(group, set()).add(subset)

    blocking_issue = None
    if projection_issue:
        blocking_issue = projection_issue
    elif not plan.rows:
        blocking_issue = "清洗后没有可用图片。请检查排除示例或扩大批次范围"
    elif plan.format_name == "yolo" and (plan.output_task or plan.project.task_type) == "detect" and polygons:
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
        "augmented_images": plan.augmented_images,
        "cleaning": cleaning_summary(plan.cleaning) if plan.cleaning is not None else None,
        "preview_token": export_fingerprint(plan, fingerprints),
    }


def export_project(store, project_id, options):
    with store.lock:
        plan, fingerprints = validated_export_plan(store, project_id, options)
        return write_archive(store, plan, fingerprints)


def validated_export_plan(store, project_id, options, *, checkpoint=None):
    plan = prepare_export(store, project_id, options, checkpoint=checkpoint)
    fingerprints = {}
    expected = options.get("preview_token")
    for row in plan.rows:
        if checkpoint:
            checkpoint()
        if "version" not in row:
            try:
                fingerprints[row["id"]] = digest(plan.project.item(row["id"])["source"])
            except OSError as exc:
                raise Problem(f"{row['photo_id']} 的原标注无法读取，请重新扫描数据", 409) from exc
    if expected is not None and expected != export_fingerprint(plan, fingerprints):
        raise Problem("数据已变化，请查看更新后的预检结果再导出", 409)
    return plan, fingerprints


def write_archive(store, plan, fingerprints):
    project_id, format_name = plan.project.id, plan.format_name
    folder = store.home / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    export_id = uuid.uuid4().hex
    path = folder / f"{project_id}_{format_name}_{export_id}.zip"
    temporary = path.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
            manifest = write_members(store, plan, archive, fingerprints)
        temporary.replace(path)
        return {"id": path.stem, "filename": path.name, "images": len(manifest["items"]), "split_counts": manifest["split_counts"],
                "original_images": len(plan.rows), "augmented_images": plan.augmented_images,
                "source_split_counts": plan.split_counts, "integrity_check": manifest.get("integrity_check"),
                "url": f"/api/exports/{path.stem}"}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class DatasetFolder:
    def __init__(self, root):
        self.root = root
        self.members = []
        self.member_names = set()
        self.directories = {root}
        root.mkdir(parents=True)

    def open(self, name, mode):
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "\\" in name or name in self.member_names:
            raise Problem("导出文件路径无效或重复，已中止导出", 422)
        path = self.root / name
        if path.parent not in self.directories:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.directories.add(path.parent)
        self.members.append(name)
        self.member_names.add(name)
        return path.open(mode + "b")

    def writestr(self, name, content):
        with self.open(name, "w") as destination:
            destination.write(content if isinstance(content, bytes) else content.encode("utf-8"))

    def write(self, source, name):
        with Path(source).open("rb") as original, self.open(name, "w") as destination:
            shutil.copyfileobj(original, destination)

    def namelist(self):
        return list(self.members)

    def read(self, name):
        return (self.root / name).read_bytes()


def write_verified_image(archive, source, name, expected_digest):
    hasher = hashlib.sha256()
    with Path(source).open("rb") as original, archive.open(name, "w") as destination:
        for block in iter(lambda: original.read(1024 * 1024), b""):
            hasher.update(block)
            destination.write(block)
    if hasher.hexdigest() != expected_digest:
        raise Problem("复制图片时原图发生变化，请重新准备数据", 409)


def labelme_shapes(shapes):
    result = []
    for shape in shapes:
        if shape["shape_type"] == "rectangle" and len(shape["points"]) == 4:
            xs, ys = zip(*shape["points"])
            shape = {**shape, "points": [[min(xs), min(ys)], [max(xs), max(ys)]]}
        result.append(shape)
    return result


def yolo_values(shape, width, height):
    if shape["shape_type"] == "polygon":
        return [coord / size for x, y in shape["points"] for coord, size in ((x, width), (y, height))]
    xs, ys = zip(*shape["points"])
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    values = [(left + right) / (2 * width), (top + bottom) / (2 * height),
              (right - left) / width, (bottom - top) / height]
    if shape["shape_type"] == "hand":
        values.extend(coord for x, y, visibility in shape["keypoints"] for coord in (x / width, y / height, visibility))
    return values


def write_annotation(archive, format_name, project, document, image_name, subset, split_config, coco_documents, image_index):
    suffix = f"/{subset}" if split_config else ""
    base_name = Path(image_name).stem
    width, height = document["imageWidth"], document["imageHeight"]
    if format_name == "labelme":
        document["imagePath"] = ("../../" if split_config else "../") + image_name
        document["imageData"] = None
        document["shapes"] = labelme_shapes(document["shapes"])
        archive.writestr(f"annotations{suffix}/{base_name}.json", json.dumps(document, ensure_ascii=False, indent=2))
    elif format_name == "yolo":
        lines = []
        for shape in document["shapes"]:
            if project.task_type == "detect" and shape["shape_type"] != "rectangle":
                raise Problem(YOLO_POLYGON_ERROR, 422)
            values = yolo_values(shape, width, height)
            lines.append(str(project.names.index(shape["label"])) + " " + " ".join(f"{value:.12f}" for value in values))
        archive.writestr(f"labels{suffix}/{base_name}.txt", "\n".join(lines) + ("\n" if lines else ""))
    elif format_name == "mask":
        mask = semantic_mask(document["shapes"], project.names, width, height)
        with io.BytesIO() as buffer:
            mask.save(buffer, format="PNG")
            archive.writestr(f"masks{suffix}/{base_name}.png", buffer.getvalue())
    else:
        coco = coco_documents[subset]
        coco["images"].append({"id": image_index, "file_name": image_name,
                               "width": document["imageWidth"], "height": document["imageHeight"]})
        for shape in document["shapes"]:
            xs, ys = zip(*shape["points"])
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            polygon = shape["points"]
            is_polygon = shape["shape_type"] == "polygon"
            entry = {"id": len(coco["annotations"]) + 1, "image_id": image_index,
                     "category_id": project.names.index(shape["label"]) + 1,
                     "bbox": [min(xs), min(ys), width, height],
                     "area": polygon_area(polygon) if is_polygon else width * height, "iscrowd": 0,
                     "segmentation": [[value for point in polygon for value in point]] if is_polygon else []}
            if shape["shape_type"] == "hand":
                entry["keypoints"] = [coord for point in shape["keypoints"] for coord in point]
                entry["num_keypoints"] = sum(point[2] > 0 for point in shape["keypoints"])
            coco["annotations"].append(entry)


def verify_export_annotation(archive, format_name, project, document, image_name, subset, split_config,
                             coco_documents, image_index, annotation_offset):
    def reject(reason):
        raise Problem(f"导出标注回读校验失败：{reason}；已中止导出", 422)

    suffix = f"/{subset}" if split_config else ""
    base_name = Path(image_name).stem
    shapes = document["shapes"]
    width, height = document["imageWidth"], document["imageHeight"]
    if format_name == "labelme":
        decoded = json.loads(archive.read(f"annotations{suffix}/{base_name}.json"))
        if (decoded.get("shapes") != labelme_shapes(shapes) or decoded.get("imageWidth") != document["imageWidth"]
                or decoded.get("imageHeight") != document["imageHeight"]
                or decoded.get("imagePath") != ("../../" if split_config else "../") + image_name):
            reject("Labelme 对象、尺寸或图片配对不一致")
    elif format_name == "yolo":
        lines = archive.read(f"labels{suffix}/{base_name}.txt").decode().splitlines()
        if len(lines) != len(shapes):
            reject("YOLO 对象数量不一致")
        for line, shape in zip(lines, shapes):
            values = line.split()
            expected = yolo_values(shape, width, height)
            try:
                if len(values) != len(expected) + 1 or int(values[0]) != project.names.index(shape["label"]):
                    reject("YOLO 类别或字段不一致")
                actual = list(map(float, values[1:]))
            except ValueError as error:
                raise Problem("YOLO 导出坐标无法解析，已中止导出", 422) from error
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)):
                reject("YOLO 坐标与原标注不一致")
    elif format_name == "mask":
        expected = semantic_mask(shapes, project.names, width, height)
        with Image.open(io.BytesIO(archive.read(f"masks{suffix}/{base_name}.png"))) as actual:
            if actual.mode != "L" or actual.size != expected.size or actual.tobytes() != expected.tobytes():
                reject("语义掩膜像素或尺寸不一致")
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
            if shape["shape_type"] == "hand" and (
                    actual.get("keypoints") != [coord for point in shape["keypoints"] for coord in point]
                    or actual.get("num_keypoints") != sum(point[2] > 0 for point in shape["keypoints"])):
                reject("COCO 关键点或可见性不一致")


def checked_variants(document, source_id, source_sha256, settings, variants, checkpoint=None):
    produced = 0
    for index, contents, augmented, provenance in variants:
        if checkpoint:
            checkpoint()
        if (index != produced + 1 or index > settings["copies"] or provenance.get("variant") != index
                or provenance.get("source_id") != source_id or provenance.get("source_sha256") != source_sha256):
            raise Problem("增强副本编号或来源不一致，已中止导出", 422)
        produced += 1
        shape_count = validate_augmented_sample(document, augmented, contents, settings, provenance)
        yield settings, index, contents, augmented, provenance, shape_count
    if produced != settings["copies"]:
        raise Problem("增强副本数量缺失，已中止导出", 422)


def prepare_export_rows(store, plan, fingerprints, checkpoint, acceleration):
    def inputs():
        for row in plan.rows:
            if checkpoint:
                checkpoint()
            current = store.detail(plan.project.id, row["id"])
            if current["version"] != row.get("version", 0) or current["status"] != row["status"]:
                raise Problem("导出期间标注发生变化，请重试", 409)
            if row["id"] in fingerprints and current["source_fingerprint"] != fingerprints[row["id"]]:
                raise Problem("导出期间原标注发生变化，请重试", 409)
            if plan.cleaning is not None and current["image_sha256"] != row["image_sha256"]:
                raise Problem("图像在清洗后发生变化，请重新预检", 409)
            document = copy.deepcopy(current["document"])
            validate_export_document(store, plan.project, document, current["width"], current["height"], current["status"])
            yield row, current, output_document(plan, document)

    check = acceleration.checkpoint if acceleration else checkpoint

    def prepare(sample):
        row, current, document = sample
        if check:
            check()
        source_contents = None
        if plan.include_images and (plan.augmentation or acceleration):
            source_contents = Path(plan.project.item(row["id"])["path"]).read_bytes()
            if hashlib.sha256(source_contents).hexdigest() != current["image_sha256"]:
                raise Problem("复制图片时原图发生变化，请重新准备数据", 409)
        variants = []
        if plan.augmentation and plan.assignments[row["id"]] == "train":
            augmentation = plan.augmentation
            settings_list = [{**augmentation, "preset": preset} for preset in augmentation["presets"]]
            cache = acceleration.cache if acceleration else None
            key = cache_key(plan.project.id, row["id"], current["image_sha256"], document, augmentation) if cache else None
            cached = cache.load(key) if cache else None
            if cached is not None:
                try:
                    if len(cached) != augmentation["variants_per_image"]:
                        raise Problem("缓存副本数量不匹配", 422)
                    for offset, settings in enumerate(settings_list):
                        start = offset * settings["copies"]
                        variants.extend(checked_variants(document, row["id"], current["image_sha256"], settings,
                                                         cached[start:start + settings["copies"]], check))
                except (Problem, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError):
                    if check:
                        check()
                    cache.discard(key)
                    variants = []
                else:
                    cache.record(True, len(variants))
                    return row, current, document, source_contents, variants
            with augmentation_image(io.BytesIO(source_contents)) as original:
                for settings in settings_list:
                    generated = augmented_variants(original, document, row["id"], current["image_sha256"], settings)
                    variants.extend(checked_variants(document, row["id"], current["image_sha256"], settings, generated, check))
            if cache:
                cache.record(False, len(variants))
                cache.save(key, variants)
        return row, current, document, source_contents, variants

    return acceleration.map(prepare, inputs()) if acceleration else map(prepare, inputs())


def write_members(store, plan, archive, fingerprints, *, checkpoint=None, progress=None, verify=False, acceleration=None):
    project, format_name, rows = output_project(plan), plan.format_name, plan.rows
    split_config = plan.split_config
    split_counts = copy.deepcopy(plan.split_counts)
    augmentation, augmented_images = plan.augmentation, plan.augmented_images
    if augmentation:
        split_counts["train"] += augmented_images
    unit = split_config["unit"] if split_config else None
    project_id = project.id
    manifest = {"project": project_id, "task_type": project.task_type, "source_task_type": plan.project.task_type,
                "format": format_name, "created_at": now(),
                "reviewed_only": plan.reviewed_only, "include_images": plan.include_images,
                "classes": project.names, "split": {"image": "random_by_image", "source_group": "grouped_by_source",
                                                     "directory": "grouped_by_directory"}.get(unit, "unassigned"),
                "split_config": split_config, "split_counts": split_counts, "source_split_counts": plan.split_counts,
                "original_images": len(rows), "augmented_images": augmented_images, "augmentation": augmentation,
                "source_writes": 0, "items": [], "warning": "评估请使用人工审核的样本。" +
                {"image": "按图片随机划分，同源 / 视频连续帧及重复图片可能跨集",
                 "source_group": "按任务清单来源组划分，实际比例取决于各组图量",
                 "directory": "按原图文件夹划分，分散在不同文件夹的同源图片可能跨集"}.get(unit, "未划分 train/val/test")}
    subsets = [name for name in SPLIT_NAMES if split_counts[name]] if split_config else ["unassigned"]
    categories = [{"id": index + 1, "name": label} for index, label in enumerate(project.names)]
    if project.task_type == "hand_pose":
        for category in categories:
            category.update(keypoints=[name for name, _ in HAND_KEYPOINTS],
                            skeleton=[[start + 1, end + 1] for start, end in HAND_SKELETON])
    coco_documents = {name: {"images": [], "annotations": [], "categories": categories} for name in subsets}
    verify_annotations = verify or augmentation or project.task_type != "detect"
    if plan.cleaning is not None:
        manifest["cleaning"] = plan.cleaning
    for row, current, document, source_contents, variants in prepare_export_rows(store, plan, fingerprints, checkpoint, acceleration):
        if checkpoint:
            checkpoint()
        item_id = row["id"]
        item = project.item(row["id"])
        safe_photo_id = re.sub(r"[^\w.-]", "_", row["photo_id"])
        filename = f"{safe_photo_id}_{row['id'][:8]}{Path(item['path']).suffix.lower()}"
        subset = plan.assignments[item_id]
        suffix = f"/{subset}" if split_config else ""
        image_name = f"images{suffix}/{filename}"
        manifest["items"].append({"photo_id": row["photo_id"], "id": item_id, "status": current["status"],
                                   "revision": current["version"], "sha256": current["image_sha256"],
                                   "source_group": item["source_group"], "image": image_name, "split": subset})
        group = split_group(row, unit)
        if group is not None:
            manifest["items"][-1]["split_group"] = group
        if plan.include_images:
            if source_contents is not None:
                archive.writestr(image_name, source_contents)
            elif verify or plan.cleaning is not None:
                write_verified_image(archive, item["path"], image_name, current["image_sha256"])
            else:
                archive.write(item["path"], image_name)
        source_document = copy.deepcopy(document) if verify_annotations else None
        annotation_offset = len(coco_documents[subset]["annotations"])
        write_annotation(archive, format_name, project, document, image_name, subset, split_config,
                         coco_documents, len(manifest["items"]))
        if verify_annotations:
            verify_export_annotation(archive, format_name, project, source_document, image_name, subset,
                                     split_config, coco_documents, len(manifest["items"]), annotation_offset)
        if augmentation:
            if hashlib.sha256(archive.read(image_name)).hexdigest() != current["image_sha256"]:
                raise Problem("原图副本指纹与已保存标注不一致，已中止导出", 409)
        if verify or augmentation:
            manifest["items"][-1]["annotation_count"] = len(source_document["shapes"])
        if progress:
            progress(len(manifest["items"]), len(rows) + augmented_images, row["photo_id"])
        for settings, index, contents, augmented, provenance, shape_count in variants:
            if checkpoint:
                checkpoint()
            preset = settings["preset"]
            variant_id = f"{row['id']}__aug_{preset}_{index}"
            variant_name = f"images/train/{Path(filename).stem}__aug_{preset}_{index}.jpg"
            archive.writestr(variant_name, contents)
            if hashlib.sha256(archive.read(variant_name)).hexdigest() != augmented["labelpilot"]["image_sha256"]:
                raise Problem("增强图片回读指纹不匹配，已中止导出", 422)
            manifest["items"].append({"photo_id": row["photo_id"], "id": variant_id, "status": "augmented",
                                      "revision": 0, "sha256": augmented["labelpilot"]["image_sha256"],
                                      "source_group": item["source_group"], "image": variant_name,
                                      "split": "train", "source_id": row["id"],
                                      "source_revision": current["version"], "human_reviewed": False,
                                      "annotation_count": shape_count, "augmentation": provenance})
            expected_document = copy.deepcopy(augmented)
            annotation_offset = len(coco_documents["train"]["annotations"])
            write_annotation(archive, format_name, project, augmented, variant_name, "train", split_config,
                             coco_documents, len(manifest["items"]))
            verify_export_annotation(archive, format_name, project, expected_document, variant_name, "train",
                                     split_config, coco_documents, len(manifest["items"]), annotation_offset)
            if progress:
                progress(len(manifest["items"]), len(rows) + augmented_images, variant_id)
    names = model_labels(project.task_type, project.names)
    archive.writestr("classes.txt", "\n".join(names) + "\n")
    if format_name in ("yolo", "mask"):
        paths = "" if not split_config else "".join(
            f"{name}: " + (f"images/{name}" if split_counts[name] else "null") + "\n" for name in SPLIT_NAMES)
        if format_name == "mask":
            paths += "masks_dir: masks\n"
            manifest["mask_classes"] = names
        elif project.task_type == "hand_pose":
            paths += f"kpt_shape: [{HAND_COUNT}, 3]\nflip_idx: {list(range(HAND_COUNT))}\n"
        archive.writestr("dataset.yaml", paths + "names:\n" + "\n".join(
            f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names)) + "\n")
    if format_name == "coco":
        for name, document in coco_documents.items():
            filename = f"annotations/{name}.json" if split_config else "annotations.json"
            archive.writestr(filename, json.dumps(document, ensure_ascii=False))
            if verify_annotations and json.loads(archive.read(filename)) != document:
                raise Problem("COCO 回读与已校验标注不一致，已中止导出", 422)
    if verify or augmentation:
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
    if acceleration:
        manifest["acceleration"] = {"workers": acceleration.workers, "cache_enabled": acceleration.cache is not None,
                                    **(acceleration.cache.stats() if acceleration.cache else {})}
    archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
