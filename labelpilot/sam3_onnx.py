import math
from pathlib import Path


MODEL_FILES = ("sam3_image_encoder.onnx", "sam3_language_encoder.onnx", "sam3_decoder.onnx")


def graph_tensors(graph):
    import onnx

    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from graph_tensors(nested)


def validate_bundle(directory):
    import onnx

    root = Path(directory).resolve(strict=True)
    for name in (*MODEL_FILES, "tokenizer.json"):
        if not (root / name).is_file():
            raise ValueError(f"SAM3 ONNX 文件缺失：{name}")
    for name in MODEL_FILES:
        model = onnx.load_model(str(root / name), load_external_data=False)
        for tensor in graph_tensors(model.graph):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            data = {entry.key: entry.value for entry in tensor.external_data}
            location = data.get("location", "")
            external = (root / location).resolve()
            if not location or not external.is_relative_to(root) or not external.is_file():
                raise ValueError(f"SAM3 ONNX 外部数据缺失或越界：{location}")
            offset = int(data.get("offset", "0"))
            length = int(data.get("length", math.prod(tensor.dims) *
                                  onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize))
            if offset < 0 or length < 0 or offset + length > external.stat().st_size:
                raise ValueError(f"SAM3 ONNX 外部数据不完整：{location}，至少需要 {offset + length} 字节")
    return root


def prepare_image(image, shape, tensor_type):
    import cv2
    import numpy as np

    if len(shape) not in (3, 4) or not all(isinstance(size, int) and size > 0 for size in shape[-2:]):
        raise ValueError("SAM3 图像编码器输入尺寸不受支持")
    pixels = np.asarray(image.convert("RGB"))
    pixels = cv2.resize(pixels, (shape[-1], shape[-2]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    if tensor_type == "tensor(float)":
        pixels = (pixels.astype(np.float32) / 255.0 - 0.5) / 0.5
    elif tensor_type != "tensor(uint8)":
        raise ValueError("SAM3 图像编码器输入类型不受支持")
    if len(shape) == 4:
        pixels = pixels[None]
    return np.ascontiguousarray(pixels)


def text_tokens(tokenizer, prompt):
    import numpy as np

    identifiers = tokenizer.encode(prompt).ids
    if len(identifiers) > 32:
        raise ValueError("SAM3 对象描述过长，请缩短至 30 个 CLIP 词元以内")
    tokens = np.zeros((1, 32), dtype=np.int64)
    tokens[0, :len(identifiers)] = identifiers
    return tokens


class Sam3OnnxRuntime:
    def __init__(self, directory, device):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.root = validate_bundle(directory)
        self.device = device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("所选 SAM3 环境无法使用 CUDA")
                ort.preload_dlls()
            except ImportError as error:
                raise RuntimeError("CUDA ONNX 环境需安装兼容的 PyTorch / CUDA DLL") from error
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("所选环境缺少 ONNX Runtime CUDA 支持，未回退到 CPU")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        if self.tokenizer.token_to_id("<|startoftext|>") != 49406 or self.tokenizer.token_to_id("<|endoftext|>") != 49407:
            raise ValueError("请选择匹配 SAM3 的 CLIP tokenizer.json")
        self.sessions = {}
        for name in MODEL_FILES:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.enable_mem_pattern = False
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.root / name), sess_options=options, providers=providers)
            session.disable_fallback()
            if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError(f"{name} 未成功启用 CUDA，已拒绝 CPU 回退")
            self.sessions[name] = session

    def predict(self, image_path, prompt, threshold):
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as source:
            if source.getexif().get(274, 1) != 1:
                raise ValueError("原图包含 EXIF 旋转，已拒绝应用以避免坐标偏移")
            image = source.convert("RGB")
        language = self.sessions["sam3_language_encoder.onnx"]
        language_outputs = dict(zip((value.name for value in language.get_outputs()),
                                    language.run(None, {"tokens": text_tokens(self.tokenizer, prompt)})))
        encoder = self.sessions["sam3_image_encoder.onnx"]
        image_input = encoder.get_inputs()[0]
        pixels = prepare_image(image, image_input.shape, image_input.type)
        image_outputs = dict(zip((value.name for value in encoder.get_outputs()),
                                encoder.run(None, {image_input.name: pixels})))
        decoder = self.sessions["sam3_decoder.onnx"]
        inputs = {**image_outputs, "original_height": np.array(image.height, dtype=np.int64),
                  "original_width": np.array(image.width, dtype=np.int64),
                  "language_mask": language_outputs["text_attention_mask"],
                  "language_features": language_outputs["text_memory"],
                  "language_embeds": language_outputs["text_embeds"],
                  "box_coords": np.zeros((1, 1, 4), dtype=np.float32),
                  "box_labels": np.ones((1, 1), dtype=np.int64),
                  "box_masks": np.ones((1, 1), dtype=np.bool_)}
        output = dict(zip((value.name for value in decoder.get_outputs()),
                          decoder.run(None, {value.name: inputs[value.name] for value in decoder.get_inputs()})))
        return output["masks"], output["scores"], image.width, image.height
