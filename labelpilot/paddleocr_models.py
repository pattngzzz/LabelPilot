import hashlib
import re
from pathlib import Path


MODEL_ROLES = {"detection": "text_detection_model_dir", "recognition": "text_recognition_model_dir"}
DEFAULT_MODELS = {"detection": "PP-OCRv5_server_det", "recognition": "PP-OCRv5_server_rec"}


def model_files(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("请选择 PaddleOCR 推理模型文件夹")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("OCR 模型必须使用完整绝对路径")
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError("请选择包含推理文件的文件夹，不是训练检查点")
        files = [path / "inference.yml", path / "inference.pdiparams"]
        graph = next((path / name for name in ("inference.json", "inference.pdmodel") if (path / name).is_file()), None)
        if graph is None or any(not file.is_file() or file.stat().st_size == 0 for file in [*files, graph]):
            raise ValueError("OCR 推理目录需要 inference.yml、inference.pdiparams 和 inference.json / inference.pdmodel；训练 .pdparams 或 YOLO .pt 不能直接使用，请先导出推理模型")
        return [*files, graph]
    except OSError as exc:
        raise ValueError("OCR 模型文件不存在或无权读取") from exc


def model_signature(value):
    files = model_files(value)
    return str(files[0].parent), tuple((file.name, file.stat().st_size, file.stat().st_mtime_ns) for file in files)


def inspect_model(value, role):
    import yaml

    if role not in MODEL_ROLES:
        raise ValueError("未知 OCR 模型类型")
    files = model_files(value)
    try:
        configuration = yaml.safe_load(files[0].read_text(encoding="utf-8"))
        name = configuration["Global"]["model_name"]
    except (KeyError, TypeError, OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("OCR inference.yml 缺少有效的 Global.model_name") from exc
    token = "det" if role == "detection" else "rec"
    if not isinstance(name, str) or not re.search(rf"(?:^|_){token}(?:_|$)", name):
        raise ValueError("检测和识别模型不能混用；请选择对应的 _det / _rec 推理模型")
    fingerprints = {}
    for file in files:
        fingerprint = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                fingerprint.update(block)
        fingerprints[file.name] = fingerprint.hexdigest()
    return {"name": name, "directory": str(files[0].parent), "sha256": fingerprints}
