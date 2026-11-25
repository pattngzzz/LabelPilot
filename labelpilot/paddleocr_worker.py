import contextlib
import json
import os
import sys
import traceback

import cv2
import numpy as np
from PIL import Image


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main():
    from paddleocr import PaddleOCR

    device = os.environ.get("LABELPILOT_OCR_DEVICE", "gpu:0")
    engine = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if request.get("command") == "shutdown":
            return
        if request.get("command") != "predict":
            continue
        response = {"id": request.get("id"), "ok": False, "models_ready": False}
        try:
            with Image.open(request["image_path"]) as image:
                image.load()
                image_array = np.asarray(image.convert("RGB")).copy()
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            with contextlib.redirect_stdout(sys.stderr):
                if engine is None:
                    engine = PaddleOCR(
                        text_detection_model_name="PP-OCRv5_server_det",
                        text_recognition_model_name="PP-OCRv5_server_rec",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        device=device,
                    )
                results = engine.predict(
                    input=image_array,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=float(request.get("text_det_thresh", 0.3)),
                )
            result = next(iter(results), None)
            detections = []
            if result is not None:
                polygons = json_value(result.get("dt_polys", []))
                texts = result.get("rec_texts", [])
                scores = json_value(result.get("rec_scores", []))
                for index, polygon in enumerate(polygons):
                    detections.append({
                        "points": polygon,
                        "text": texts[index] if index < len(texts) else "",
                        "score": scores[index] if index < len(scores) else 0.0,
                    })
            response.update(ok=True, models_ready=True, detections=detections,
                            runtime={"device": device, "model": "PP-OCRv5_server_det+PP-OCRv5_server_rec"})
        except Exception as exc:
            response["error"] = str(exc)[:500]
            response["traceback"] = traceback.format_exc()[-2000:]
        sys.stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
