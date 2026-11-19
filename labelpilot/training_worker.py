import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from .inference import DISABLED_INTEGRATIONS, model_class_names, validate_model_task
from .store import Problem, atomic_json, read_json
from .tasks import MODEL_TASK, TRAIN_MODEL, model_labels


def metric_keys(task):
    if task == "semantic":
        return {"miou": "metrics/mIoU", "pixel_accuracy": "metrics/pixel_acc"}
    kind = {"detect": "B", "segment": "M", "hand_pose": "P"}[task]
    return {name: f"metrics/{column}({kind})" for name, column in
            (("precision", "precision"), ("recall", "recall"), ("map50", "mAP50"), ("map50_95", "mAP50-95"))}


def watch_parent(parent_pid, stop_file):
    if os.name == "nt":
        from ctypes import WinDLL, wintypes

        kernel32 = WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE
        if handle:
            try:
                while not stop_file.exists() and kernel32.WaitForSingleObject(handle, 250) == 258:  # WAIT_TIMEOUT
                    pass
            finally:
                kernel32.CloseHandle(handle)
    else:
        while os.getppid() == parent_pid and not stop_file.exists():
            time.sleep(0.25)
    os._exit(130)


def train(run):
    config = read_json(run / "train.json")
    options = config["options"]
    task = config.get("task_type", "detect")
    runtime = run / "ultralytics"
    runtime.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(run / "matplotlib"))
    os.environ.update(YOLO_CONFIG_DIR=str(runtime),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", NO_ALBUMENTATIONS_UPDATE="1")

    print("加载本地 PyTorch…", flush=True)
    import torch
    print("初始化 Ultralytics…", flush=True)
    from matplotlib import get_data_path
    from ultralytics import YOLO, __version__
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR

    SETTINGS.update({name: False for name in DISABLED_INTEGRATIONS if name in SETTINGS})
    # Dataset validation looks for these font names even when plots are disabled.
    fallback = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        shutil.copyfile(fallback, USER_CONFIG_DIR / name)

    cuda = torch.cuda.is_available()
    if options["device"] == "cuda" and not cuda:
        raise Problem("当前 PyTorch 环境没有可用 CUDA。请选择 CPU 或切换到 CUDA 环境")
    use_cuda = cuda and options["device"] != "cpu"
    starting = run / "starting.pt"
    fine_tune = options["mode"] == "fine_tune"
    if fine_tune and not starting.is_file():
        raise Problem("缺少起始权重副本")
    try:
        model = YOLO(str(starting) if fine_tune else TRAIN_MODEL[task])
    except FileNotFoundError as exc:
        if task != "semantic" or fine_tune:
            raise
        raise Problem("语义分割需要包含 YOLO26 semantic 模型的 Ultralytics 版本") from exc
    if model.task != MODEL_TASK[task]:
        raise Problem(f"训练需要 {MODEL_TASK[task]} 权重，当前模型为 {model.task}")
    if fine_tune:
        validate_model_task(model, task)
        model_class_names(model.names, model_labels(task, config["classes"]), config.get("label_aliases"))

    progress = {"epoch": 0, "epochs": options["epochs"], "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
                "ultralytics": __version__, "torch": str(torch.__version__), "metrics": {}}
    atomic_json(run / "progress.json", progress)

    metrics = metric_keys(task)

    def report(trainer):
        progress["epoch"] = min(options["epochs"], trainer.epoch + 1)
        for name, key in metrics.items():
            value = trainer.metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                progress["metrics"][name] = float(value)
        atomic_json(run / "progress.json", progress)

    model.add_callback("on_fit_epoch_end", report)
    model.train(data=str(run / "dataset" / "dataset.yaml"), project=str(run), name="run", exist_ok=False,
                epochs=options["epochs"], imgsz=options["image_size"], batch=options["batch_size"],
                seed=options["seed"], device=0 if use_cuda else "cpu", pretrained=fine_tune,
                workers=0, cache=False, plots=False, amp=False, save=True, val=True)


if __name__ == "__main__":
    job = Path(sys.argv[1]).resolve()
    threading.Thread(target=watch_parent, args=(int(sys.argv[2]), job / "cancel.request"), daemon=True).start()
    try:
        train(job)
    except Exception as exc:
        atomic_json(job / "error.json", {"error": str(getattr(exc, "message", exc))[:500]})
        raise
