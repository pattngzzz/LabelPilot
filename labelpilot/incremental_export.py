import copy
import hashlib
import json
import random
import stat
import threading
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path, PurePosixPath

from .store import Problem, digest, now
from .tasks import model_labels


class ExportFileCache:
    def __init__(self, limit=50000):
        self.limit = limit
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def key(self, path):
        state = path.stat()
        return (path.absolute(), state.st_size, state.st_mtime_ns, state.st_ctime_ns, state.st_dev, state.st_ino)

    def digest(self, path, checkpoint=None):
        path = Path(path)
        if checkpoint:
            checkpoint()
        key = self.key(path)
        with self.lock:
            cached = self.entries.get(key)
            if cached is not None:
                self.entries.move_to_end(key)
                return cached
        result = digest(path, checkpoint=checkpoint)
        if self.key(path) != key:
            raise Problem("文件在校验过程中发生变化，请等待文件稳定后刷新预检", 409)
        with self.lock:
            self.entries[key] = result
            self.entries.move_to_end(key)
            while len(self.entries) > self.limit:
                self.entries.popitem(last=False)
        return result


def content_hash(document):
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def annotation_content_hash(document):
    ignored = {"score", "prediction_origin", "prediction_id", "text_recognition_confidence"}
    return content_hash({"imageWidth": document["imageWidth"], "imageHeight": document["imageHeight"],
                         "shapes": [{key: value for key, value in shape.items() if key not in ignored}
                                    for shape in document["shapes"]]})


def source_references(entry, project_id):
    return copy.deepcopy(entry.get("sources") or [{"project": entry.get("source_project", project_id),
        "item": entry.get("source_item", entry["id"]), "revision": entry["revision"]}])


def file_stamp(path):
    state = path.stat()
    return {"size": state.st_size, "mtime_ns": state.st_mtime_ns}


def safe_path(root, name, checked_directories=None):
    relative = PurePosixPath(name)
    if (not relative.parts or relative.is_absolute() or relative.as_posix() != name or "\\" in name
            or any(part in (".", "..") or ":" in part or part.rstrip(" .") != part for part in relative.parts)
            or Path(name).is_reserved()):
        raise Problem("增量数据集清单包含不安全的文件路径", 422)
    path = root.joinpath(*relative.parts)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if checked_directories is not None and parent in checked_directories:
            continue
        try:
            state = parent.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(state.st_mode) or getattr(state, "st_file_attributes", 0) & 0x400:
            raise Problem("增量数据集内不能包含符号链接或目录联接", 422)
        if checked_directories is not None and stat.S_ISDIR(state.st_mode):
            checked_directories.add(parent)
    return path


def entry_files(entry, manifest):
    if entry.get("files"):
        return list(entry["files"])
    names = [entry["image"]] if manifest["include_images"] else []
    image = PurePosixPath(entry["image"])
    format_name = manifest["format"]
    directory, extension = {"yolo": ("labels", ".txt"), "labelme": ("annotations", ".json"),
                            "mask": ("masks", ".png"), "paddleocr": ("annotations", ".json")}.get(format_name, (None, None))
    if directory:
        names.append(str(PurePosixPath(directory, *image.parts[1:]).with_suffix(extension)))
    return names


def record_entry(archive, entry, offset, source=None, document=None, source_stat=None):
    if not hasattr(archive, "root"):
        return
    entry["files"] = {}
    for name in archive.members[offset:]:
        path = archive.root / name
        entry["files"][name] = {**file_stamp(path), "sha256": digest(path)}
        if name == entry["image"] and entry["files"][name]["sha256"] != entry["sha256"]:
            raise Problem("原图副本回读指纹不一致，已中止导出", 409)
    if source is not None:
        entry["source_stat"] = source_stat or file_stamp(Path(source))
        entry["annotation_sha256"] = content_hash(document)


def load_incremental(project, format_name, options, *, checkpoint=None, progress=None, file_cache=None):
    enabled = options.get("incremental", False)
    if type(enabled) is not bool:
        raise Problem("增量导出开关必须为布尔值", 422)
    if not enabled:
        return None
    merge_projects = options.get("merge_projects", False)
    if type(merge_projects) is not bool:
        raise Problem("跨项目合并开关必须为布尔值", 422)
    from .model_files import directory

    root = directory(options.get("destination"))
    if options.get("clean", False):
        raise Problem("增量导出不能同时清洗；请取消清洗，或完整导出到新目录", 422)
    manifest_path = safe_path(root, "export_manifest.json")
    if not manifest_path.exists():
        if any((root / name).exists() for name in ("images", "labels", "annotations", "masks", "classes.txt", "dataset.yaml")):
            raise Problem("所选目录存在数据集文件但缺少 export_manifest.json，无法安全增量合并；请完整导出到新目录", 422)
        return None
    try:
        state = IncrementalExport(root, manifest_path.read_bytes())
        state.checkpoint = checkpoint
        state.progress = progress
        state.file_cache = file_cache
        manifest = state.manifest
        if (manifest["project"] != project.id and not merge_projects or manifest["format"] != format_name
                or manifest["task_type"] != (options.get("output_task") or project.task_type)
                or manifest["include_images"] != options.get("include_images", True)):
            raise Problem("目标数据集的项目、格式、标注用途或包含原图设置不一致，请保持一致或完整导出到新目录", 422)
        state.cross_project = manifest["project"] != project.id
        state.multi_project = state.cross_project or len(manifest.get("source_projects", [])) > 1
        if state.multi_project and (format_name != "paddleocr" or project.task_type != "ocr"
                or manifest.get("source_task_type", manifest["task_type"]) != "ocr"
                or not options.get("reviewed_only", True) or not manifest.get("reviewed_only")
                or any(entry["status"] != "reviewed" for entry in manifest["items"] if "source_id" not in entry)):
            raise Problem("跨项目合并仅支持已人工审核、包含原图的 PaddleOCR 数据集", 422)
        classes = manifest["classes"]
        if (not isinstance(classes, list) or not classes or len(set(classes)) != len(classes)
                or any(not isinstance(name, str) or not name.strip() for name in classes)):
            raise ValueError("classes")
        if safe_path(root, "classes.txt").read_text(encoding="utf-8-sig").splitlines() != model_labels(manifest["task_type"], classes):
            raise Problem("目标 classes.txt 与清单类别顺序不一致，不能安全合并", 422)
        state.validate_entries()
        return state
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Problem(f"目标导出清单或目录结构不完整，无法安全增量导出：{exc}", 422) from exc


class IncrementalExport:
    def __init__(self, root, contents):
        self.root = root
        self.manifest = json.loads(contents)
        self.baseline_hash = hashlib.sha256(contents).hexdigest()
        self.originals = {}
        self.by_source = {}
        self.stamps = {}
        self.changed = set()
        self.added = self.updated = self.skipped = 0
        self.old_files = set()
        self.source_stamps = {}
        self.checked_directories = set()
        self.augmentation_changed = False
        self.new_classes = []
        self.checkpoint = None
        self.progress = None
        self.file_cache = None
        self.cross_project = False
        self.multi_project = False
        self.references = {}
        self.deduplicated_rows = set()
        self.deduplicated = 0

    def export_id(self, row):
        return row.get("export_id", row["id"])

    def bind_sources(self, store, project, rows, checkpoint):
        from .exporting import split_group, validate_export_document

        identities, hashes = {}, {}
        for item_id, entry in self.originals.items():
            references = source_references(entry, self.manifest["project"])
            self.references[item_id] = references
            for reference in references:
                identity = (reference["project"], reference["item"])
                if identity in identities and identities[identity] != item_id:
                    raise Problem("旧清单的来源身份重复，不能安全跨项目合并", 422)
                identities[identity] = item_id
            hashes.setdefault(entry["sha256"], []).append(item_id)
        pending, incoming_signatures, selected = {}, {}, []
        unit = (self.manifest.get("split_config") or {}).get("unit", "image")
        for row in rows:
            if checkpoint:
                checkpoint()
            document = row["document"]
            validate_export_document(store, project, document, document["imageWidth"], document["imageHeight"], row["status"])
            source = Path(project.item(row["id"])["path"])
            stamp = file_stamp(source)
            image_hash = self.digest(source)
            if file_stamp(source) != stamp or image_hash != document.get("labelpilot", {}).get("image_sha256"):
                raise Problem(f"{row['photo_id']} 原图已改变，请重新审核后导出", 409)
            self.source_stamps[source] = stamp
            signature = annotation_content_hash(document)
            identity = (project.id, row["id"])
            item_id = identities.get(identity)
            matches = hashes.get(image_hash, [])
            if len(matches) > 1:
                raise Problem(f"{row['photo_id']} 在旧数据集中已有多个同图样本，不能安全跨项目去重", 409)
            previous = self.originals.get(item_id)
            if previous and previous["sha256"] != image_hash:
                raise Problem(f"{row['photo_id']} 的来源图片内容已改变，拒绝覆盖历史数据", 409)
            if item_id is None and matches:
                item_id = matches[0]
                previous = self.originals[item_id]
            if previous:
                annotation = next(name for name in entry_files(previous, self.manifest) if name.startswith("annotations/"))
                if not self.check_file(annotation, previous.get("files", {}).get(annotation)):
                    raise Problem(f"历史标注文件缺失或损坏：{annotation}；请先修复原数据集", 409)
                stored = json.loads(safe_path(self.root, annotation).read_bytes())
                if item_id in incoming_signatures and incoming_signatures[item_id] != signature:
                    raise Problem(f"同图标注冲突：{row['photo_id']} 与本批其他来源的标注不同", 409)
                incoming_signatures[item_id] = signature
                same_annotation = annotation_content_hash(stored) == signature
                primary = (previous.get("source_project", self.manifest["project"]), previous.get("source_item", previous["id"]))
                shared = len(self.references[item_id]) > 1
                if not same_annotation and (identity != primary or shared):
                    raise Problem(f"同图标注冲突：{row['photo_id']} 的原文、坐标或属性与已导出版本不同；未覆盖历史标注，请先统一标注或导出到新目录", 409)
                if unit != "image" and split_group(row, unit) != previous.get("split_group"):
                    raise Problem(f"同图来源分组不一致：{row['photo_id']}；请统一来源组后合并", 409)
                if identity != primary:
                    self.deduplicated_rows.add(row["id"])
                    self.deduplicated += 1
            elif image_hash in pending:
                item_id, previous_signature, group = pending[image_hash]
                if previous_signature != signature:
                    raise Problem(f"同图标注冲突：{row['photo_id']} 与本批其他图片内容相同但标注不同", 409)
                if unit != "image" and split_group(row, unit) != group:
                    raise Problem(f"同图来源分组不一致：{row['photo_id']}；请统一来源组后合并", 409)
                self.skipped += 1
                self.deduplicated += 1
                self.references[item_id].append({"project": project.id, "item": row["id"], "revision": row.get("version", 0)})
                continue
            else:
                item_id = hashlib.sha256(f"{project.id}:{row['id']}".encode()).hexdigest()
                if item_id in self.originals or item_id in self.references:
                    raise Problem("跨项目来源编号冲突，已停止合并", 409)
                self.references[item_id] = []
                pending[image_hash] = (item_id, signature, split_group(row, unit))
            reference = {"project": project.id, "item": row["id"], "revision": row.get("version", 0)}
            references = self.references[item_id]
            references[:] = [entry for entry in references if (entry["project"], entry["item"]) != identity]
            references.append(reference)
            selected.append({**row, "export_id": item_id})
        return selected

    def digest(self, path):
        if self.file_cache is not None:
            return self.file_cache.digest(path, self.checkpoint)
        return digest(path, checkpoint=self.checkpoint)

    def validate_entries(self):
        manifest = self.manifest
        if manifest["format"] == "paddleocr":
            legacy = {PurePosixPath(entry["image"]).with_suffix("").as_posix().replace("images/", "recognition/", 1): entry
                      for entry in manifest["items"] if not entry.get("files")}
            if legacy:
                subsets = ("train", "val", "test") if manifest["split_config"] else ("all",)
                for subset in subsets:
                    path = safe_path(self.root, f"rec/{subset}.txt")
                    if not path.exists():
                        continue
                    for line in path.read_text(encoding="utf-8").splitlines():
                        name = line.split("\t", 1)[0]
                        owner = legacy.get(name.rsplit("_", 1)[0])
                        if not owner:
                            raise ValueError("旧 OCR 识别清单存在无法配对的裁剪")
                        if not owner.get("files"):
                            owner["files"] = {member: {} for member in entry_files(owner, manifest)}
                        owner["files"][name] = {}
        ids, images, files = set(), set(), set()
        for index, entry in enumerate(manifest["items"]):
            if self.checkpoint:
                self.checkpoint()
            if self.progress:
                self.progress("检查旧数据集结构", index, len(manifest["items"]), entry.get("image", ""))
            item_id, image = entry["id"], entry["image"]
            if (not isinstance(item_id, str) or item_id in ids or image.casefold() in images
                    or entry["split"] not in ("train", "val", "test", "unassigned")
                    or not image.startswith("images/")):
                raise ValueError("重复图片或无效的图片条目")
            ids.add(item_id)
            images.add(image.casefold())
            expected_parent = PurePosixPath("images", entry["split"]) if manifest["split_config"] else PurePosixPath("images")
            if PurePosixPath(image).parent != expected_parent:
                raise ValueError("图片目录与集合归属不一致")
            source_id = entry.get("source_id", item_id)
            self.by_source.setdefault(source_id, []).append(entry)
            if "source_id" not in entry:
                self.originals[item_id] = entry
                references = source_references(entry, manifest["project"])
                identities = set()
                for reference in references:
                    identity = (reference["project"], reference["item"])
                    if (any(not isinstance(value, str) or not value for value in identity)
                            or type(reference["revision"]) is not int or reference["revision"] < 0 or identity in identities):
                        raise ValueError("无效或重复的来源记录")
                    identities.add(identity)
                if (entry.get("source_project", manifest["project"]), entry.get("source_item", item_id)) not in identities:
                    raise ValueError("来源记录缺少原始标注身份")
            for name in entry_files(entry, manifest):
                safe_path(self.root, name, self.checked_directories)
                image_path, member = PurePosixPath(image), PurePosixPath(name)
                expected = [image] if manifest["include_images"] else []
                directory, extension = {"yolo": ("labels", ".txt"), "labelme": ("annotations", ".json"),
                                        "mask": ("masks", ".png"), "paddleocr": ("annotations", ".json")}.get(manifest["format"], (None, None))
                if directory:
                    expected.append(str(PurePosixPath(directory, *image_path.parts[1:]).with_suffix(extension)))
                crop = (manifest["format"] == "paddleocr" and member.parent == PurePosixPath("recognition", *image_path.parts[1:-1])
                        and member.suffix == ".png" and member.stem.rsplit("_", 1)[0] == image_path.stem)
                if name.casefold() in files or name not in expected and not crop:
                    raise ValueError("文件清单包含冲突或不属于该图片的文件")
                files.add(name.casefold())
                self.old_files.add(name)
            if entry.get("files") and not set(expected).issubset(entry["files"]):
                raise ValueError("文件清单缺少图片或配对标注")
        if any(source_id not in self.originals for source_id in self.by_source):
            raise ValueError("增强图片缺少对应原图")
        if (manifest["original_images"] != len(self.originals)
                or manifest["augmented_images"] != len(manifest["items"]) - len(self.originals)):
            raise ValueError("清单图片数量不一致")
        if manifest["split_config"] and any(manifest["split_counts"][subset] != sum(entry["split"] == subset for entry in manifest["items"])
                                            for subset in ("train", "val", "test")):
            raise ValueError("清单集合数量不一致")
        for name in self.config_files():
            path = safe_path(self.root, name, self.checked_directories)
            if not path.is_file():
                raise ValueError(f"缺少 {name}")
            self.stamps[name] = {**file_stamp(path), "sha256": self.digest(path)}
            self.old_files.add(name)
            recorded = manifest.get("metadata_files", {}).get(name)
            if recorded and recorded["sha256"] != self.stamps[name]["sha256"]:
                raise Problem(f"目标配置或汇总标注被外部修改，拒绝覆盖：{name}；请完整导出到新目录", 409)

    def config_files(self):
        manifest = self.manifest
        names = ["classes.txt", "export_manifest.json"]
        if manifest["format"] in ("yolo", "mask"):
            names.append("dataset.yaml")
        subsets = ("train", "val", "test") if manifest["split_config"] else ("unassigned",)
        if manifest["format"] == "coco":
            names.extend(f"annotations/{subset}.json" if manifest["split_config"] else "annotations.json"
                         for subset in subsets if any(entry["split"] == subset for entry in manifest["items"]))
        if manifest["format"] == "paddleocr":
            names.append("OCR_README.txt")
            names.extend(f"{directory}/{subset if subset != 'unassigned' else 'all'}.txt"
                         for subset in subsets for directory in ("det", "rec")
                         if any(entry["split"] == subset for entry in manifest["items"]))
        return names

    def assignments(self, rows, settings, ratios, seed, unit):
        from .exporting import SPLIT_NAMES, split_group

        old = self.manifest["split_config"]
        if bool(settings) != bool(old) or settings and unit != old["unit"]:
            raise Problem("增量导出不能改变是否划分或划分单位；请保持一致或完整导出到新目录", 422)
        if not settings:
            return {row["id"]: "unassigned" for row in rows}, None, None
        counts = Counter(entry["split"] for entry in self.originals.values())
        active = [name for name in SPLIT_NAMES if ratios[name] > 0]
        groups = {}
        if unit != "image":
            for entry in self.originals.values():
                group = entry.get("split_group")
                if group is None:
                    raise Problem("旧清单缺少来源划分依据，不能安全增量合并，请完整导出到新目录", 422)
                if group in groups and groups[group] != entry["split"]:
                    raise Problem("旧数据同一来源已跨集合，不能安全继续分组增量导出", 422)
                groups[group] = entry["split"]
        assignments, pending = {}, {}
        for row in rows:
            previous = self.originals.get(self.export_id(row))
            group = split_group(row, unit)
            if unit != "image" and group is None:
                raise Problem("增量图片缺少来源组，请补齐来源信息", 422)
            if previous:
                if unit != "image" and previous.get("split_group") != group:
                    raise Problem("已导出图片的来源分组发生变化，请完整导出到新目录", 422)
                assignments[row["id"]] = previous["split"]
            elif group in groups:
                assignments[row["id"]] = groups[group]
                counts[groups[group]] += 1
            else:
                pending.setdefault(group if unit != "image" else row["id"], []).append(row["id"])
        ordered = sorted(pending)
        random.Random(seed).shuffle(ordered)
        ordered.sort(key=lambda group: len(pending[group]), reverse=True)
        total = len(self.originals) + sum(self.export_id(row) not in self.originals for row in rows)
        for group in ordered:
            subset = max(active, key=lambda name: total * ratios[name] / 100 - counts[name])
            for item_id in pending[group]:
                assignments[item_id] = subset
            counts[subset] += len(pending[group])
        config = {**ratios, "seed": seed, "unit": unit,
                  "method": "seeded_shuffle" if unit == "image" else "seeded_group_balance"}
        return assignments, config, {name: sum(subset == name for subset in assignments.values()) for name in SPLIT_NAMES}

    def check_file(self, name, expected=None):
        path = safe_path(self.root, name, self.checked_directories)
        if not path.is_file():
            return False
        stamp = file_stamp(path)
        if expected and all(stamp[key] == expected.get(key) for key in stamp):
            self.stamps[name] = expected
            return True
        actual = self.digest(path)
        self.stamps[name] = {**stamp, "sha256": actual}
        return not expected or actual == expected.get("sha256")

    def reusable(self, plan, row, document, checkpoint):
        from .exporting import output_document, output_project, verify_export_annotation

        previous = self.originals.get(self.export_id(row))
        if not previous or (previous["revision"], previous["status"]) != (row.get("version", 0), row["status"]):
            return False
        signature = content_hash(output_document(plan, document))
        if previous.get("annotation_sha256") and previous["annotation_sha256"] != signature:
            return False
        source = Path(plan.project.item(row["id"])["path"])
        self.source_stamps[source] = file_stamp(source)
        if previous.get("source_stat") != file_stamp(source):
            if self.digest(source) != previous["sha256"]:
                raise Problem(f"{row['photo_id']} 原图已改变，已阻止旧标注套用", 409)
        if plan.assignments[row["id"]] == "train" and self.manifest.get("augmentation") != plan.augmentation:
            return False
        if not previous.get("annotation_sha256"):
            if plan.format_name not in ("yolo", "labelme", "mask") or not self.manifest.get("integrity_check", {}).get("passed"):
                return False
            reader = type("Reader", (), {"read": lambda reader, name: safe_path(self.root, name).read_bytes()})()
            try:
                for entry in self.by_source[self.export_id(row)]:
                    if checkpoint:
                        checkpoint()
                    expected = output_document(plan, document)
                    if "source_id" in entry:
                        provenance = entry["augmentation"]
                        if entry["source_revision"] != previous["revision"] or provenance["source_sha256"] != previous["sha256"]:
                            return False
                        width, height = expected["imageWidth"], expected["imageHeight"]
                        operations = provenance["operations"]
                        angle = operations.get("rotation_degrees", 0)
                        if angle not in (0, 90, 180, 270):
                            return False
                        for shape in expected["shapes"]:
                            points = shape["points"]
                            if operations["horizontal_flip"]:
                                points = [[width - point[0], point[1]] for point in points]
                            if angle == 90:
                                points = [[point[1], width - point[0]] for point in points]
                            elif angle == 180:
                                points = [[width - point[0], height - point[1]] for point in points]
                            elif angle == 270:
                                points = [[height - point[1], point[0]] for point in points]
                            shape["points"] = points
                        if angle in (90, 270):
                            expected["imageWidth"], expected["imageHeight"] = height, width
                    verify_export_annotation(reader, plan.format_name, output_project(plan), expected,
                                             entry["image"], entry["split"], plan.split_config, {}, 0, 0)
            except (Problem, OSError, ValueError):
                return False
        for entry in self.by_source[self.export_id(row)]:
            files = {}
            for name in entry_files(entry, self.manifest):
                if checkpoint:
                    checkpoint()
                expected = entry.get("files", {}).get(name)
                if not self.check_file(name, expected):
                    return False
                if name == entry["image"] and self.stamps[name]["sha256"] != entry["sha256"]:
                    return False
                files[name] = self.stamps[name]
            entry["files"] = files
        previous.update(annotation_sha256=signature, source_stat=file_stamp(source))
        return True

    def select(self, store, plan, checkpoint):
        self.new_classes = [name for name in plan.project.names if name not in self.manifest["classes"]]
        self.augmentation_changed = self.manifest.get("augmentation") != plan.augmentation
        names = self.manifest["classes"] + [name for name in plan.project.names if name not in self.manifest["classes"]]
        if plan.project.task_type == "semantic" and len(names) > 254:
            raise Problem("合并历史类别后超过语义掩膜类别上限，请完整导出到新目录", 422)
        project = copy.copy(plan.project)
        project.names = names
        if self.multi_project:
            project.export_sources = [config for config in store.configs if config["id"] == self.manifest["project"]]
        plan = replace(plan, project=project, incremental=self)
        if (self.manifest.get("augmentation") != plan.augmentation
                and set(self.originals) - {self.export_id(row) for row in plan.rows}):
            raise Problem("增强配置发生变化，需选择包含所有历史原图的范围才能同步；或完整导出到新目录", 422)
        changed = []
        selected_ids = {self.export_id(row) for row in plan.rows if row["id"] not in self.deduplicated_rows}
        outside = set(self.originals) - selected_ids
        total = len(outside) + len(plan.rows)
        for index, source_id in enumerate(sorted(outside)):
            if self.progress:
                self.progress("核验历史图片与增强文件", index, total, self.originals[source_id].get("photo_id", source_id))
            for entry in self.by_source[source_id]:
                files = {}
                for name in entry_files(entry, self.manifest):
                    if checkpoint:
                        checkpoint()
                    if (not self.check_file(name, entry.get("files", {}).get(name))
                            or name == entry["image"] and self.stamps[name]["sha256"] != entry["sha256"]):
                        raise Problem("范围外历史文件已缺失或损坏，请扩大导出范围以修复，或完整导出到新目录", 409)
                    files[name] = self.stamps[name]
                entry["files"] = files
        for index, row in enumerate(plan.rows, len(outside)):
            if checkpoint:
                checkpoint()
            if self.progress:
                self.progress("核验历史图片与增强文件", index, total, row["photo_id"])
            document = row.get("document")
            if document is None:
                document = store.detail(project.id, row["id"])["document"]
            if row["id"] in self.deduplicated_rows or self.reusable(plan, row, document, checkpoint):
                self.skipped += 1
            else:
                self.changed.add(self.export_id(row))
                changed.append(row)
                if self.export_id(row) in self.originals:
                    self.updated += 1
                else:
                    self.added += 1
        if self.progress:
            self.progress("核验历史图片与增强文件", total, total, "")
        counts = ({name: sum(plan.assignments[row["id"]] == name for row in changed) for name in ("train", "val", "test")}
                  if plan.split_config else None)
        return replace(plan, rows=changed, split_counts=counts)

    def summary(self):
        return {"directory": str(self.root), "added": self.added, "updated": self.updated, "skipped": self.skipped,
                "cross_project": self.cross_project, "multi_project": self.multi_project, "deduplicated": self.deduplicated,
                "target_project": self.manifest["project"], "settings_preserved": self.multi_project,
                "augmentation_changed": self.augmentation_changed, "new_classes": self.new_classes,
                "existing_originals": len(self.originals), "existing_assignments": "preserved",
                "split_policy": "比例和种子仅用于新增图片，旧图保持集合归属；不自动删除范围外历史数据"}

    def merge(self, plan, dataset, manifest):
        from .exporting import write_configuration

        retained = [entry for entry in self.manifest["items"] if entry.get("source_id", entry["id"]) not in self.changed]
        manifest["items"] = retained + manifest["items"]
        if self.multi_project:
            manifest["project"] = self.manifest["project"]
            sources = {entry["id"]: entry for entry in self.manifest.get("source_projects", [])}
            for config in plan.project.config, *plan.project.export_sources:
                sources[config["id"]] = {"id": config["id"], "name": config["name"], "image_root": config["image_root"]}
            sources.setdefault(manifest["project"], {"id": manifest["project"], "name": manifest["project"]})
            manifest["source_projects"] = list(sources.values())
            for entry in manifest["items"]:
                source_id = entry.get("source_id", entry["id"])
                original = self.originals.get(source_id, entry)
                entry.setdefault("source_project", original.get("source_project", self.manifest["project"]))
                entry.setdefault("source_item", original.get("source_item", source_id))
                if "source_id" not in entry:
                    entry["sources"] = self.references[entry["id"]]
        images, files = set(), set()
        for entry in manifest["items"]:
            image = entry["image"].casefold()
            if image in images:
                raise Problem("增量图片文件名与历史图片冲突，已停止合并", 409)
            images.add(image)
            for name in entry_files(entry, manifest):
                if name.casefold() in files:
                    raise Problem("增量标注或裁剪文件名与历史文件冲突，已停止合并", 409)
                files.add(name.casefold())
        originals = [entry for entry in manifest["items"] if "source_id" not in entry]
        manifest.update(created_at=self.manifest["created_at"], updated_at=now(), incremental=self.summary(),
                        original_images=len(originals), augmented_images=len(manifest["items"]) - len(originals))
        if plan.split_config:
            manifest["split_counts"] = {name: sum(entry["split"] == name for entry in manifest["items"]) for name in ("train", "val", "test")}
            manifest["source_split_counts"] = {name: sum(entry["split"] == name for entry in originals) for name in ("train", "val", "test")}
        manifest["integrity_check"]["scope"] = "new_and_updated_files; retained_files_checked_against_manifest"
        manifest["integrity_check"]["retained_images"] = len(retained)
        self.merge_aggregates(dataset, manifest, retained)
        write_configuration(dataset, plan, manifest, rewrite=True)
        dataset.rewrite("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    def merge_aggregates(self, dataset, manifest, retained):
        if manifest["format"] not in ("coco", "paddleocr"):
            return
        subsets = ("train", "val", "test") if manifest["split_config"] else ("unassigned",)
        for subset in subsets:
            kept = {entry["image"] for entry in retained if entry["split"] == subset}
            if manifest["format"] == "coco":
                name = f"annotations/{subset}.json" if manifest["split_config"] else "annotations.json"
                old_path, new_path = safe_path(self.root, name), dataset.root / name
                documents = []
                if old_path.exists():
                    documents.append((json.loads(old_path.read_bytes()), kept))
                if new_path.exists():
                    documents.append((json.loads(new_path.read_bytes()), None))
                if not documents:
                    continue
                merged = {"images": [], "annotations": [], "categories": copy.deepcopy(documents[-1][0]["categories"])}
                merged["categories"] = [{**merged["categories"][0], "id": index + 1, "name": label}
                                        for index, label in enumerate(manifest["classes"])]
                for document, allowed in documents:
                    mapping = {}
                    for image in document["images"]:
                        if allowed is None or image["file_name"] in allowed:
                            mapping[image["id"]] = len(merged["images"]) + 1
                            merged["images"].append({**image, "id": mapping[image["id"]]})
                    for annotation in document["annotations"]:
                        if annotation["image_id"] in mapping:
                            merged["annotations"].append({**annotation, "id": len(merged["annotations"]) + 1,
                                                           "image_id": mapping[annotation["image_id"]]})
                dataset.rewrite(name, json.dumps(merged, ensure_ascii=False))
            else:
                label = subset if subset != "unassigned" else "all"
                stems = {PurePosixPath(image).stem for image in kept}
                for directory in ("det", "rec"):
                    name = f"{directory}/{label}.txt"
                    old_path, new_path = safe_path(self.root, name), dataset.root / name
                    lines = []
                    if old_path.exists():
                        for line in old_path.read_text(encoding="utf-8").splitlines():
                            image = line.split("\t", 1)[0]
                            keep = image in kept if directory == "det" else PurePosixPath(image).stem.rsplit("_", 1)[0] in stems
                            if keep:
                                lines.append(line)
                    if new_path.exists():
                        lines.extend(new_path.read_text(encoding="utf-8").splitlines())
                    if not lines and not any(entry["split"] == subset for entry in manifest["items"]):
                        continue
                    dataset.rewrite(name, "".join(line + "\n" for line in lines))
                    manifest["ocr"]["splits"].setdefault(label, {})["images" if directory == "det" else "recognition_crops"] = len(lines)

    def publish(self, dataset, checkpoint):
        if digest(safe_path(self.root, "export_manifest.json")) != self.baseline_hash:
            raise Problem("目标数据集在导出期间发生变化，请重新预检", 409)
        new_files = set(dataset.namelist())
        removed = {name for source_id in self.changed for entry in self.by_source.get(source_id, [])
                   for name in entry_files(entry, self.manifest)} - new_files
        targets = new_files | removed
        for path, expected in self.source_stamps.items():
            if not path.is_file() or file_stamp(path) != expected:
                raise Problem("跳过的原图在导出期间发生变化，请重新预检", 409)
        before = {}
        for name in targets:
            path = safe_path(self.root, name)
            if path.exists() and name not in self.old_files:
                raise Problem(f"目标已有未登记文件，拒绝覆盖：{name}", 409)
            if path.exists():
                before[name] = {**file_stamp(path), "sha256": digest(path)}
        for name, expected in self.stamps.items():
            path = safe_path(self.root, name)
            if not path.is_file() or file_stamp(path) != {key: expected[key] for key in ("size", "mtime_ns")}:
                raise Problem("目标数据集文件在导出期间发生变化，请重新预检", 409)
        new_files = {name for name in new_files if name not in before or digest(dataset.root / name) != before[name]["sha256"]}
        targets = new_files | removed
        backup = dataset.root / ".rollback"
        operations = []
        try:
            ordered = sorted(targets - {"export_manifest.json"}) + ["export_manifest.json"]
            for name in ordered:
                checkpoint()
                destination = safe_path(self.root, name)
                if name in before:
                    if not destination.is_file() or digest(destination) != before[name]["sha256"]:
                        raise Problem("待更新文件在合并期间发生变化，已停止并恢复", 409)
                elif destination.exists():
                    raise Problem("合并期间出现文件冲突，已停止并恢复", 409)
                destination.parent.mkdir(parents=True, exist_ok=True)
                previous = backup / name
                had_previous = destination.exists()
                if had_previous:
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(previous)
                operations.append((name, had_previous))
                if name in new_files:
                    (dataset.root / name).replace(destination)
        except Exception:
            try:
                for name, had_previous in reversed(operations):
                    destination = safe_path(self.root, name)
                    destination.unlink(missing_ok=True)
                    if had_previous:
                        (backup / name).replace(destination)
            except Exception as exc:
                self.recovery_required = True
                raise Problem(f"增量合并中断且自动恢复失败；请保留恢复目录并人工恢复：{dataset.root}；{exc}", 500) from exc
            raise


def record_metadata(dataset, manifest):
    manifest["metadata_files"] = {}
    for name in dataset.members:
        if name != "export_manifest.json" and not name.startswith(("images/", "labels/", "masks/", "recognition/")):
            if name.startswith("annotations/") and name.removeprefix("annotations/") not in ("train.json", "val.json", "test.json"):
                continue
            path = dataset.root / name
            manifest["metadata_files"][name] = {**file_stamp(path), "sha256": digest(path)}
    dataset.rewrite("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
