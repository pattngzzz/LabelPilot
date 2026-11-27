import os
import stat
import time
from pathlib import Path

from .store import Problem


MAX_ENTRIES = 20000
MAX_MODELS = 200
SEARCH_SECONDS = 5


def directory(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Problem("请选择本机文件夹", 422)
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise Problem("请使用文件夹的完整绝对路径", 422)
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise Problem("请选择文件夹，而不是文件", 422)
    except (OSError, RuntimeError, ValueError):
        raise Problem("文件夹不存在或无权访问，请选择其他位置", 422)
    return path


def roots():
    if os.name == "nt":
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        return [f"{chr(65 + index)}:/" for index in range(26) if mask & (1 << index)]
    return ["/"]


def linked(entry):
    if entry.is_symlink():
        return True
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def weight(entry):
    info = entry.stat(follow_symlinks=False)
    return {"name": entry.name, "path": str(Path(entry.path)), "size": info.st_size}


def browse(path, suffixes=(".pt",), *, names=None):
    path = directory(path)
    folders, files = [], []
    partial = False
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_ENTRIES:
                    partial = True
                    break
                try:
                    if linked(entry):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        folders.append({"name": entry.name, "path": str(Path(entry.path))})
                    elif ((entry.name in names if names is not None else Path(entry.name).suffix.lower() in suffixes)
                          and entry.is_file(follow_symlinks=False)):
                        files.append(weight(entry))
                except OSError:
                    partial = True
    except OSError:
        raise Problem("无法读取此文件夹，请检查权限或选择其他位置", 422)
    return {"path": str(path), "parent": str(path.parent) if path.parent != path else None,
            "roots": roots(), "directories": sorted(folders, key=lambda item: item["name"].lower()),
            "files": sorted(files, key=lambda item: item["name"].lower()), "partial": partial, "search": False}


def search(path):
    path = directory(path)
    pending = [path]
    files = []
    partial = False
    visited = 0
    deadline = time.monotonic() + SEARCH_SECONDS
    while pending and visited < MAX_ENTRIES and len(files) < MAX_MODELS and time.monotonic() < deadline:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if visited >= MAX_ENTRIES or len(files) >= MAX_MODELS or time.monotonic() >= deadline:
                        partial = True
                        break
                    visited += 1
                    try:
                        if linked(entry):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.name.lower().endswith(".pt") and entry.is_file(follow_symlinks=False):
                            model = weight(entry)
                            model["name"] = str(Path(entry.path).relative_to(path))
                            files.append(model)
                    except OSError:
                        partial = True
        except OSError:
            partial = True
    return {"path": str(path), "parent": str(path.parent) if path.parent != path else None,
            "roots": roots(), "directories": [], "files": sorted(files, key=lambda item: item["name"].lower()),
            "partial": partial or bool(pending), "search": True}
