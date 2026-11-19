import json

from PIL import Image

from .store import Problem


def annotation_key(document, *, ordered=False):
    shapes = []
    for original in document["shapes"]:
        shape = {name: value for name, value in original.items()
                 if name not in ("score", "prediction_id", "prediction_origin")}
        if shape["shape_type"] == "rectangle":
            xs, ys = zip(*shape["points"])
            shape["points"] = [[min(xs), min(ys)], [max(xs), max(ys)]]
        shape["points"] = [[float(x), float(y)] for x, y in shape["points"]]
        shapes.append(json.dumps(shape, sort_keys=True, ensure_ascii=False))
    return tuple(shapes if ordered else sorted(shapes))


def clean_selection(store, project, rows, *, include_documents=False, checkpoint=None):
    by_digest = {}
    retained = {}
    excluded = []
    for row in rows:
        if checkpoint:
            checkpoint()
        identity = {"id": row["id"], "photo_id": row["photo_id"]}
        try:
            current = store.detail(project.id, row["id"], verify_image=True)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            excluded.append({**identity, "reason": "unreadable", "detail": str(exc)[:180]})
            continue
        except Problem as exc:
            excluded.append({**identity, "reason": "unreadable", "detail": exc.message})
            continue
        if current["version"] != row["version"] or current["status"] != "reviewed":
            raise Problem("清洗期间审核记录发生变化，请刷新预检", 409)
        document = current["document"]
        try:
            store.validate(project, document, current["width"], current["height"], "reviewed",
                           document.get("labelpilot", {}).get("negative_sample_confirmed", False))
        except Problem as exc:
            excluded.append({**identity, "reason": "annotation", "detail": exc.message})
            continue
        verified = {**row, "image_sha256": current["image_sha256"]}
        if include_documents:
            verified["document"] = document
        by_digest.setdefault(current["image_sha256"], []).append(
            (verified, annotation_key(document, ordered=project.task_type == "semantic")))

    for copies in by_digest.values():
        kept, first_key = copies[0]
        if all(key == first_key for _, key in copies):
            retained[kept["id"]] = kept
            for duplicate, _ in copies[1:]:
                excluded.append({"id": duplicate["id"], "photo_id": duplicate["photo_id"], "reason": "duplicate",
                                 "kept_id": kept["id"], "detail": f"保留 {kept['photo_id']}"})
        else:
            for conflicting, _ in copies:
                excluded.append({"id": conflicting["id"], "photo_id": conflicting["photo_id"], "reason": "conflict",
                                 "detail": "字节相同的图片存在不同标注"})
    return [retained[row["id"]] for row in rows if row["id"] in retained], excluded
