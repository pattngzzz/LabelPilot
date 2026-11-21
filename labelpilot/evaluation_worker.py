import gc
import io
import math
import os
import sys
import threading
from pathlib import Path

from PIL import Image

from .blindsets import frozen_image_bytes, frozen_image_path, read_snapshot, verified_manifest
from .comparison_details import record_frame, write_details
from .evaluation import comparison_class_names, count_detections
from .inference import DISABLED_INTEGRATIONS
from .store import Problem, atomic_json, digest, read_annotation_snapshot, read_json
from .training_worker import watch_parent


def frozen_frames(directory, snapshot, manifest):
    root = (directory / "dataset").resolve()
    frames = []
    for item in manifest["items"]:
        image = frozen_image_path(directory, item)
        annotation = f"annotations/{image.stem}.json"
        label_path = (root / annotation).resolve()
        if not label_path.is_relative_to(root / "annotations"):
            raise Problem("盲测快照路径无效", 409)
        document, fingerprint = read_annotation_snapshot(label_path)
        if fingerprint != snapshot["annotation_sha256"].get(annotation):
            raise Problem(f"{item['photo_id']} 的盲测标注与冻结版本不一致", 409)
        labels = []
        for shape in document["shapes"]:
            xs, ys = zip(*shape["points"])
            labels.append((shape["label"], (min(xs), min(ys), max(xs), max(ys))))
        frames.append((image, item, labels, (document["imageWidth"], document["imageHeight"])))
    return frames


def predict_frame(model, names, image_path, item, size, options, device):
    contents = frozen_image_bytes(image_path, item)
    with Image.open(io.BytesIO(contents)) as image:
        output = model.predict(image.convert("RGB"), conf=options["confidence"], iou=options["nms_iou"],
                               imgsz=options["image_size"], device=device, max_det=options["max_det"],
                               rect=False, verbose=False, save=False)[0]
    width, height = size
    if tuple(output.orig_shape) != (height, width):
        raise Problem("预测图片尺寸与盲测快照不一致")
    detections = []
    for values in output.boxes.data.detach().cpu().tolist():
        left, top, right, bottom, score, class_id = values
        if not all(math.isfinite(value) for value in values) or not 0 <= class_id < len(names) or int(class_id) != class_id:
            raise Problem("模型返回了无效的检测框或类别")
        box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        if box[0] < box[2] and box[1] < box[3] and score >= options["confidence"]:
            detections.append((names[int(class_id)], score, box))
    return detections


def compare(run, directory, inputs):
    setup = read_json(run / "compare.json")
    options, models = setup["options"], setup["models"]
    snapshot = read_snapshot(directory)
    manifest = verified_manifest(directory, snapshot)
    classes = manifest["classes"]
    frames = frozen_frames(directory, snapshot, manifest)
    runtime = run / "ultralytics"
    runtime.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(runtime), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")

    import torch
    from ultralytics import YOLO, __version__
    from ultralytics.utils import SETTINGS

    SETTINGS.update({name: False for name in DISABLED_INTEGRATIONS if name in SETTINGS})
    device = 0 if torch.cuda.is_available() else "cpu"
    results = {}
    with write_details(run, working_directory=inputs) as details:
        for model_index, role in enumerate(("old", "new")):
            path = inputs / f"{role}.pt"
            if digest(path) != models[role]["sha256"]:
                raise Problem("评测权重副本的指纹已改变", 409)
            model = YOLO(str(path))
            if model.task != "detect":
                raise Problem(f"{models[role]['filename']} 需要是 YOLO 目标检测权重", 422)
            names = comparison_class_names(model.names, classes, setup.get("label_aliases"),
                                           setup.get("model_label_aliases"))
            models[role]["class_names"] = names
            counts = {name: dict.fromkeys(("ground_truth", "matched", "missed", "false_positives"), 0) for name in classes}
            for index, (image, item, labels, size) in enumerate(frames, 1):
                predictions = predict_frame(model, names, image, item, size, options, device)
                matches = count_detections(counts, labels, predictions, options["match_iou"])
                record_frame(details, role, index, item, labels, size, matches)
                if index in (1, len(frames)) or index % 5 == 0:
                    atomic_json(run / "progress.json", {"model": role, "done": model_index * len(frames) + index,
                                                         "total": 2 * len(frames)})
            for row in counts.values():
                row["recall"] = row["matched"] / row["ground_truth"] if row["ground_truth"] else None
            results[role] = counts
            del model
            gc.collect()
            if device == 0:
                torch.cuda.empty_cache()
    atomic_json(run / "report.json", {"snapshot": {key: snapshot[key] for key in ("id", "name", "images", "manifest_sha256")},
                                      "models": models, "options": options, "runtime": {"device": device, "ultralytics": __version__, "torch": str(torch.__version__)},
                                      "has_details": True,
                                      "classes": [{"name": name, "old": results["old"][name], "new": results["new"][name]} for name in classes]})


if __name__ == "__main__":
    job = Path(sys.argv[1]).resolve()
    threading.Thread(target=watch_parent, args=(int(sys.argv[2]), job / "cancel.request"), daemon=True).start()
    try:
        compare(job, Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve())
    except Exception as exc:
        atomic_json(job / "error.json", {"error": str(getattr(exc, "message", exc))[:500]})
        raise
