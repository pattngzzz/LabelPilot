import copy
import json
import re
import uuid
import zipfile
from pathlib import Path

from .store import Problem, now


def export_project(store, project_id, options):
    project = store.project(project_id)
    format_name = options.get("format", "yolo")
    if format_name not in ("yolo", "labelme", "coco"):
        raise Problem("导出格式必须为 yolo、labelme 或 coco", 422)
    reviewed_only = options.get("reviewed_only", True)
    if not isinstance(reviewed_only, bool) or not isinstance(options.get("include_images", True), bool):
        raise Problem("导出开关必须为布尔值", 422)
    listing = store.listing(project_id, batch=options.get("batch", ""), limit=1000000)
    rows = [item for item in listing["items"] if item["status"] == "reviewed" or
            (not reviewed_only and item["status"] in ("imported", "predicted", "draft"))]
    if not rows:
        raise Problem("没有符合条件的标注；默认只导出已由人工确认的图片", 422)
    if len(rows) > 10000:
        raise Problem("请按批次导出，每次不超过 10000 张", 422)
    folder = store.home / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    export_id = uuid.uuid4().hex
    path = folder / f"{project_id}_{format_name}_{export_id}.zip"
    manifest = {"project": project_id, "format": format_name, "created_at": now(),
                "reviewed_only": reviewed_only, "classes": project.names, "split": "unassigned",
                "source_writes": 0, "items": [], "warning": "模型预标注不等于真值；未自动划分 train/val/test"}
    coco = {"images": [], "annotations": [], "categories": [{"id": index + 1, "name": name} for index, name in enumerate(project.names)]}
    try:
        with store.lock, zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
            for image_index, row in enumerate(rows, 1):
                current = store.detail(project_id, row["id"])
                if reviewed_only and current["status"] != "reviewed":
                    raise Problem("导出期间审核状态发生变化，请重试", 409)
                document = copy.deepcopy(current["document"])
                store.validate(project, document, current["width"], current["height"], current["status"],
                               document.get("labelpilot", {}).get("negative_sample_confirmed", False))
                item = project.item(row["id"])
                safe_photo_id = re.sub(r"[^\w.-]", "_", row["photo_id"])
                filename = f"{safe_photo_id}_{row['id'][:8]}{Path(item['path']).suffix.lower()}"
                image_name = f"images/{filename}"
                base_name = Path(filename).stem
                manifest["items"].append({"photo_id": row["photo_id"], "id": row["id"], "status": current["status"],
                                           "revision": current["version"], "sha256": current["image_sha256"],
                                           "source_group": item["source_group"], "image": image_name})
                if options.get("include_images", True):
                    archive.write(item["path"], image_name)
                if format_name == "labelme":
                    document["imagePath"] = "../" + image_name
                    document["imageData"] = None
                    archive.writestr(f"annotations/{base_name}.json", json.dumps(document, ensure_ascii=False, indent=2))
                elif format_name == "yolo":
                    lines = []
                    for shape in document["shapes"]:
                        if shape["shape_type"] != "rectangle":
                            raise Problem("YOLO 检测导出不支持多边形；请改选 COCO / Labelme，避免静默转换", 422)
                        xs, ys = zip(*shape["points"])
                        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                        values = [(left + right) / 2 / current["width"], (top + bottom) / 2 / current["height"],
                                  (right - left) / current["width"], (bottom - top) / current["height"]]
                        lines.append(str(project.names.index(shape["label"])) + " " + " ".join(f"{value:.12f}" for value in values))
                    archive.writestr(f"labels/{base_name}.txt", "\n".join(lines) + ("\n" if lines else ""))
                else:
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
                archive.writestr("dataset.yaml", "names:\n" + "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(project.names)) + "\n")
            if format_name == "coco":
                archive.writestr("annotations.json", json.dumps(coco, ensure_ascii=False))
        return {"id": path.stem, "filename": path.name, "images": len(rows), "url": f"/api/exports/{path.stem}"}
    except Exception:
        path.unlink(missing_ok=True)
        raise
