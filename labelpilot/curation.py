import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .annotation_tools import box_bounds
from .evaluation import iou
from .store import Problem


SIMILARITY_LEVELS = {"strict": (4, 7), "balanced": (7, 12), "loose": (9, 19)}


def visual_signature(path):
    with Image.open(path) as original:
        width, height = original.size
        original.thumbnail((256, 256), Image.Resampling.LANCZOS)
        image = original.convert("RGB")
    grid = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = sum((grid[row * 9 + col] > grid[row * 9 + col + 1]) << (row * 8 + col)
               for row in range(8) for col in range(8))
    colors = image.resize((6, 6), Image.Resampling.BILINEAR).tobytes()
    means = tuple(sum(colors[channel::3]) / 36 for channel in range(3))
    return bits, colors, means, width, height


def hash_keys(signature, max_distance):
    # Two 64-bit hashes this close must agree in at least one of these disjoint partitions.
    parts = max_distance + 1
    for index in range(parts):
        start, end = index * 64 // parts, (index + 1) * 64 // parts
        yield index, (signature >> start) & ((1 << (end - start)) - 1)


def similar_frames(store, project_id, batch, level):
    if not isinstance(batch, str) or not isinstance(level, str) or level not in SIMILARITY_LEVELS:
        raise Problem("请选择有效的相似图范围和相似程度", 422)
    project = store.project(project_id)
    listing = store.listing(project_id, batch=batch, status="unstarted", limit=10000)
    if listing["filtered"] > 10000:
        raise Problem("本次范围超过 10000 张，请按批次筛选", 422)
    distance, color_limit = SIMILARITY_LEVELS[level]
    index = defaultdict(list)
    groups, signatures = [], []
    unreadable = 0
    for item in listing["items"]:
        path = Path(project.item(item["id"])["path"]).resolve()
        try:
            if not path.is_relative_to(project.root):
                raise OSError("image outside project")
            signature, colors, means, width, height = visual_signature(path)
        except (OSError, ValueError, Image.DecompressionBombError):
            unreadable += 1
            continue
        keys = list(hash_keys(signature, distance))
        candidates = sorted({candidate for key in keys for candidate in index.get(key, ())})
        match = None
        for candidate in candidates:
            previous, palette, other_means, other_width, other_height = signatures[candidate]
            aspect = width * other_height / (height * other_width)
            if not .88 <= aspect <= 1.14 or (signature ^ previous).bit_count() > distance:
                continue
            if sum(abs(first - second) for first, second in zip(means, other_means)) > 3 * color_limit:
                continue
            if sum(abs(first - second) for first, second in zip(colors, palette)) <= len(colors) * color_limit:
                match = candidate
                break
        member = {key: item[key] for key in ("id", "photo_id", "relative", "batch")}
        member.update(width=width, height=height)
        if match is not None:
            groups[match].append(member)
        else:
            for key in keys:
                index[key].append(len(groups))
            groups.append([member])
            signatures.append((signature, colors, means, width, height))
    excluded = store.similarity_exclusions(project_id, batch)
    matches = [{"members": group, "representative": max(group, key=lambda member: member["width"] * member["height"])["id"]}
               for group in groups if len(group) > 1]
    return {"images": listing["filtered"], "unreadable": unreadable, "groups": matches,
            "excluded": sorted(excluded), "batch": batch}


def annotation_findings(labels, predictions):
    bounds = [(shape["label"], box_bounds(shape)) for shape in labels]
    findings = []
    for prediction in predictions:
        label, box = prediction["label"], box_bounds(prediction)
        overlaps = [(iou(box, truth), name) for name, truth in bounds]
        same = max((overlap for overlap, name in overlaps if name == label), default=0)
        if same >= .5:
            continue
        overlap, name = max(overlaps, default=(0, None), key=lambda row: row[0])
        if overlap >= .5 and name != label:
            kind = "class_conflict"
        elif same >= .1:
            kind = "box_offset"
            overlap, name = same, label
        else:
            kind = "missing"
        findings.append({"kind": kind, "prediction": prediction, "annotation_label": name if overlap >= .1 else None,
                         "iou": overlap})
    return findings


def save_findings(store, project_id, item_id, before, prediction):
    with store.lock:
        current = store.detail(project_id, item_id)
        identity = ("version", "image_sha256", "source_fingerprint")
        if (any(current[key] != before[key] for key in identity) or
                (prediction["base_version"], prediction["image_sha256"]) != (current["version"], current["image_sha256"])):
            raise Problem("标注在复核扫描期间已更新，可重新扫描", 409)
        findings = annotation_findings(current["document"]["shapes"], prediction["shapes"])
        summary = {}
        if findings:
            summary = {"count": len(findings), "kinds": dict(Counter(finding["kind"] for finding in findings)),
                       "confidence": max(finding["prediction"]["score"] for finding in findings)}
        evidence = {key: current[key] for key in identity}
        evidence.update(id=uuid.uuid4().hex, findings=findings, provenance=prediction["provenance"])
        store.save_audit(project_id, item_id, evidence, summary)
    return bool(findings)
