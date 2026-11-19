import hashlib
import json
import re
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from .exporting import DatasetFolder, SPLIT_UNITS, plan_splits, prepare_export, split_group, write_members
from .store import Problem, atomic_json, digest, now, read_json, source_group


ID = re.compile(r"[0-9a-f]{32}")


def list_blindsets(store, project_id):
    store.project(project_id)
    snapshots = []
    for directory in (store.data / "blindsets").glob("*"):
        if not ID.fullmatch(directory.name) or not directory.is_dir():
            continue
        record = read_snapshot(directory)
        if record["project"] == project_id:
            snapshots.append(record)
    return sorted(snapshots, key=lambda row: row["created_at"], reverse=True)


def read_snapshot(directory):
    try:
        record = read_json(directory / "snapshot.json")
        if not isinstance(record, dict) or record.get("id") != directory.name or not record.get("project"):
            raise ValueError("missing snapshot identity")
    except (OSError, ValueError) as exc:
        raise Problem("盲测集记录无法读取，请恢复对应快照", 409) from exc
    return record


def get_blindset(store, project_id, set_id):
    if not isinstance(set_id, str) or not ID.fullmatch(set_id):
        raise Problem("盲测集不存在", 404)
    directory = store.data / "blindsets" / set_id
    if not directory.is_dir():
        raise Problem("盲测集不存在", 404)
    record = read_snapshot(directory)
    if record["project"] != project_id:
        raise Problem("盲测集不存在", 404)
    return directory, record


def verified_manifest(directory, record):
    try:
        contents = (directory / "dataset" / "export_manifest.json").read_bytes()
        if hashlib.sha256(contents).hexdigest() != record["manifest_sha256"]:
            raise ValueError("snapshot fingerprint mismatch")
        return json.loads(contents)
    except (OSError, ValueError, KeyError) as exc:
        raise Problem("盲测集清单与冻结版本不一致，请恢复对应快照", 409) from exc


def frozen_image_path(directory, item):
    root = (directory / "dataset").resolve()
    path = (root / item["image"]).resolve()
    if not path.is_relative_to(root / "images"):
        raise Problem("盲测快照图片路径无效", 409)
    return path


def frozen_image_bytes(path, item):
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise Problem(f"{item['photo_id']} 的盲测图片无法读取", 409) from exc
    if hashlib.sha256(contents).hexdigest() != item["sha256"]:
        raise Problem(f"{item['photo_id']} 的盲测图片与冻结版本不一致", 409)
    return contents


def freeze_blindset(store, project_id, payload):
    name = payload.get("name")
    percent = payload.get("percent", 20)
    seed = payload.get("seed", 42)
    project = store.project(project_id)
    unit = payload.get("unit", "source_group" if project.summary()["has_source_groups"] else "image")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64 or any(ord(char) < 32 for char in name.strip()):
        raise Problem("请输入 1 到 64 个字符的盲测集名称", 422)
    if type(percent) is not int or not 1 <= percent <= 100 or unit not in SPLIT_UNITS:
        raise Problem("冻结比例必须为 1 到 100 的整数，并选择有效划分方式", 422)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise Problem("随机种子必须为 0 到 4294967295 的整数", 422)

    with store.lock:
        plan = prepare_export(store, project_id, {"format": "labelme", "batch": payload.get("batch"), "clean": True},
                              include_documents=True)
        rows = plan.rows
        if percent < 100:
            assignments, _, _ = plan_splits(rows, {"split": {"train": 100 - percent, "val": 0, "test": percent,
                                                           "unit": unit, "seed": seed}})
            rows = [row for row in rows if assignments[row["id"]] == "test"]
        if any(shape["shape_type"] != "rectangle" for row in rows for shape in row["document"]["shapes"]):
            raise Problem("盲测集使用矩形检测框。选中的图片含多边形，请选用检测框批次", 422)
        plan = replace(plan, rows=rows, assignments=dict.fromkeys((row["id"] for row in rows), "unassigned"))
        record = {"id": uuid.uuid4().hex, "name": name.strip(), "project": project_id, "created_at": now(),
                  "batch": plan.batch, "percent": percent, "unit": unit, "seed": seed, "images": len(rows),
                  "objects": sum(row["count"] for row in rows), "confirmed_empty": sum(not row["count"] for row in rows),
                  "cleaned_out": len(plan.cleaning),
                  "source_groups": sorted({group for row in rows if (group := source_group(row["source_group"]))}),
                  "folders": sorted({split_group(row, "directory") for row in rows}) if unit == "directory" else []}
        root = store.data / "blindsets"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".freeze-") as temporary:
            staging = Path(temporary) / "snapshot"
            dataset = staging / "dataset"
            manifest = write_members(store, plan, DatasetFolder(dataset), {})
            label_entries = [f"annotations/{Path(item['image']).stem}.json" for item in manifest["items"]]
            if len(set(label_entries)) != len(rows):
                raise Problem("盲测集出现重名，请缩小批次范围", 422)
            record["annotation_sha256"] = {path: digest(dataset / path) for path in label_entries}
            record["manifest_sha256"] = digest(dataset / "export_manifest.json")
            atomic_json(staging / "snapshot.json", record)
            staging.rename(root / record["id"])
    return record


def reserve_blind_images(store, project_id, rows):
    ids, digests, groups, folders = set(), set(), set(), set()
    for record in list_blindsets(store, project_id):
        manifest = verified_manifest(store.data / "blindsets" / record["id"], record)
        ids.update(item["id"] for item in manifest["items"])
        digests.update(item["sha256"] for item in manifest["items"])
        groups.update(record["source_groups"])
        folders.update(record["folders"])
    return [row for row in rows if row["id"] not in ids and row["image_sha256"] not in digests
            and source_group(row["source_group"]) not in groups and split_group(row, "directory") not in folders]
