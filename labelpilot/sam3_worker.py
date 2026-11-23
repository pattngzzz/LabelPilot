import argparse
import json
import math
import os
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def mask_polygon(mask):
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.002), True).reshape(-1, 2)
    if len(polygon) < 3 or cv2.contourArea(polygon) < 5:
        return None
    return polygon.astype(float).tolist()


class Sam3Runtime:
    def __init__(self, checkpoint, device):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("所选环境无法使用 CUDA；请检查 PyTorch CUDA 版本和显卡驱动")
        self.device = device
        model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False,
                                       device=device, eval_mode=True, enable_segmentation=True)
        self.processor = Sam3Processor(model, device=device, confidence_threshold=0.1)

    def predict(self, image_path, prompt, threshold):
        import numpy as np
        import torch
        from PIL import Image

        path = Path(image_path).resolve(strict=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        state = self.processor.set_image(image)
        result = self.processor.set_text_prompt(prompt, state)
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or scores is None:
            return {"candidates": [], "width": image.width, "height": image.height}
        masks = masks.detach().cpu().numpy()
        scores = scores.detach().cpu().reshape(-1).tolist()
        if len(masks) != len(scores):
            raise RuntimeError("SAM3 返回的掩膜数量与分数不一致")
        return mask_candidates(masks, scores, image.width, image.height, threshold)


def mask_candidates(masks, scores, width, height, threshold):
    import numpy as np

    scores = np.asarray(scores).reshape(-1)
    if len(masks) != len(scores):
        raise RuntimeError("SAM3 返回的掩膜数量与分数不一致")
    candidates = []
    for index in np.argsort(scores)[::-1][:40]:
        score = float(scores[index])
        if not math.isfinite(score) or score < threshold:
            continue
        mask = np.asarray(masks[index])
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if tuple(mask.shape) != (height, width):
            raise RuntimeError("SAM3 掩膜尺寸与原图不一致，已拒绝应用以避免坐标偏移")
        points = mask_polygon(mask > 0)
        if points:
            candidates.append({"score": score, "points": points})
    return {"candidates": candidates, "width": width, "height": height}


class OnnxRuntime:
    def __init__(self, directory, device):
        from labelpilot.sam3_onnx import Sam3OnnxRuntime

        self.model = Sam3OnnxRuntime(directory, device)

    def predict(self, image_path, prompt, threshold):
        masks, scores, width, height = self.model.predict(image_path, prompt, threshold)
        return mask_candidates(masks, scores, width, height, threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        from labelpilot.sam3_onnx import validate_bundle

        validate_bundle(args.checkpoint)
        print("SAM3 ONNX 文件完整性检查通过")
        return
    if args.port is None:
        parser.error("--port is required for inference")
    token = os.environ.get("LABELPILOT_SAM3_TOKEN", "")
    device = os.environ.get("LABELPILOT_SAM3_DEVICE", "cuda")
    if not token:
        raise SystemExit("Missing local service token")
    runtime = (OnnxRuntime if args.backend == "onnx" else Sam3Runtime)(args.checkpoint, device)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return secrets.compare_digest(self.headers.get("X-LabelPilot-SAM3", ""), token)

        def do_GET(self):
            if not self.authorized():
                return self.respond(403, {"detail": "forbidden"})
            if self.path != "/health":
                return self.respond(404, {"detail": "not found"})
            self.respond(200, {"ready": True, "device": device, "backend": args.backend})

        def do_POST(self):
            if not self.authorized():
                return self.respond(403, {"detail": "forbidden"})
            if self.path == "/shutdown":
                self.respond(200, {"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/predict":
                return self.respond(404, {"detail": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    return self.respond(413, {"detail": "request too large"})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                image_path, prompt = payload.get("image_path"), payload.get("prompt")
                threshold = payload.get("threshold", 0.35)
                if not isinstance(image_path, str) or not isinstance(prompt, str) or not prompt.strip():
                    return self.respond(422, {"detail": "image path and prompt are required"})
                if not isinstance(threshold, (int, float)) or not 0.05 <= threshold <= 0.95:
                    return self.respond(422, {"detail": "threshold must be between 0.05 and 0.95"})
                self.respond(200, runtime.predict(image_path, prompt.strip(), float(threshold)))
            except (OSError, ValueError, TypeError) as error:
                self.respond(422, {"detail": str(error)[:300]})
            except Exception:
                traceback.print_exc()
                self.respond(500, {"detail": "SAM3 inference failed; check the local worker log"})

        def log_message(self, fmt, *values):
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()
