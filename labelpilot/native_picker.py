import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .model_files import directory
from .store import IMAGE_SUFFIXES, Problem


PICKER_LOCK = threading.Lock()
PICKER_TIMEOUT = 300


def initial_directory(value):
    if value is not None and (not isinstance(value, str) or "\x00" in value):
        raise Problem("初始文件夹路径无效", 422)
    if value and value.strip():
        try:
            path = Path(value.strip()).expanduser()
            if path.is_absolute():
                path = path.resolve(strict=True)
                if path.is_file():
                    path = path.parent
                if path.is_dir():
                    return str(path)
        except (OSError, RuntimeError, ValueError):
            pass
    return str(Path.home())


def select_images(mode, path=None):
    if mode not in ("file", "folder"):
        raise Problem("请选择图片文件或文件夹", 422)
    initial = initial_directory(path)
    if not PICKER_LOCK.acquire(blocking=False):
        raise Problem("文件选择窗口已打开，请先完成或取消选择", 409)
    try:
        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "labelpilot.native_picker"],
                input=json.dumps({"mode": mode, "path": initial}, ensure_ascii=False),
                encoding="utf-8", capture_output=True, timeout=PICKER_TIMEOUT,
                cwd=Path(__file__).resolve().parent.parent,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired:
            raise Problem("文件选择超时，窗口已关闭，请重新点击选择", 408)
        except OSError:
            raise Problem("无法启动系统文件选择窗口，请检查本机 Python 环境", 503)
        if result.returncode:
            raise Problem("无法打开系统文件选择窗口，请确认 Python 支持 Tk 且服务运行在本机桌面", 503)
        try:
            selected = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise Problem("系统文件选择窗口未返回有效路径，请重试", 503)
        if not isinstance(selected, list) or len(selected) > 10000 or any(
                not isinstance(value, str) or not value or "\x00" in value for value in selected):
            raise Problem("所选路径无效或图片超过 10000 张，请重新选择", 422)
        if not selected:
            return {"cancelled": True}
        if mode == "folder":
            if len(selected) != 1:
                raise Problem("请只选择一个图片文件夹", 422)
            return {"cancelled": False, "image_root": str(directory(selected[0])), "image_files": []}
        files = []
        for value in selected:
            try:
                path = Path(value)
                if not path.is_absolute():
                    raise ValueError()
                path = path.resolve(strict=True)
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError()
            except (OSError, RuntimeError, ValueError):
                raise Problem("所选图片不存在、格式不支持或无权访问，请重新选择", 422)
            files.append(path)
        root = files[0].parent
        if any(path.parent != root for path in files) or len(set(files)) != len(files):
            raise Problem("请选择同一文件夹内不重复的图片", 422)
        return {"cancelled": False, "image_root": str(root), "image_files": [path.name for path in files]}
    finally:
        PICKER_LOCK.release()


def show_dialog(mode, path):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.title("LabelPilot")
    root.attributes("-topmost", True)
    root.update()
    try:
        options = {"parent": root, "initialdir": path}
        if mode == "folder":
            selected = filedialog.askdirectory(title="LabelPilot · 选择待标注图片文件夹", mustexist=True, **options)
            return [selected] if selected else []
        return list(filedialog.askopenfilenames(
            title="LabelPilot · 选择待标注图片（可多选）",
            filetypes=[("图片文件", " ".join("*" + suffix for suffix in sorted(IMAGE_SUFFIXES)))], **options))
    finally:
        root.destroy()


if __name__ == "__main__":
    settings = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    selection = show_dialog(settings["mode"], settings["path"])
    sys.stdout.buffer.write(json.dumps(selection, ensure_ascii=False).encode("utf-8"))
