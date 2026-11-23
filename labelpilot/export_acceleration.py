import hashlib
import json
import os
import re
import threading
import time
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .store import Problem


CACHE_VERSION = 1
CACHE_LIMIT = 10 * 1024 ** 3
ENTRY_LIMIT = 128 * 1024 ** 2


def export_workers(value=0):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2, 4, 8):
        raise Problem("导出并行数请选择自动、1、2、4 或 8", 422)
    return value or max(1, min(4, (os.cpu_count() or 2) - 1))


def cache_key(project_id, source_id, image_sha256, document, settings):
    request = {"version": CACHE_VERSION, "project": project_id, "source": source_id,
               "image_sha256": image_sha256, "document": document, "settings": settings}
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class AugmentationCache:
    def __init__(self, root, max_bytes=CACHE_LIMIT):
        self.root = root
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.entries = {}
        self.hits = self.misses = self.errors = 0
        try:
            if root.is_symlink():
                raise OSError("cache directory is a symbolic link")
            root.mkdir(parents=True, exist_ok=True)
            for path in root.glob("*.cache"):
                if re.fullmatch(r"[0-9a-f]{64}", path.stem) and path.is_file() and not path.is_symlink():
                    info = path.stat()
                    self.entries[path.name] = (info.st_size, info.st_mtime)
            self.trim(0)
        except OSError:
            self.root = None
            self.errors += 1

    def trim(self, incoming):
        total = sum(size for size, _ in self.entries.values())
        if total + incoming <= self.max_bytes:
            return
        for name, (size, _) in sorted(self.entries.items(), key=lambda entry: entry[1][1]):
            if total + incoming <= self.max_bytes:
                break
            try:
                (self.root / name).unlink(missing_ok=True)
            except OSError:
                continue
            total -= size
            self.entries.pop(name, None)

    def discard(self, key):
        if self.root is None:
            return
        with self.lock:
            self.errors += 1
            try:
                (self.root / f"{key}.cache").unlink(missing_ok=True)
                self.entries.pop(f"{key}.cache", None)
            except OSError:
                pass

    def load(self, key):
        if self.root is None:
            return None
        path = self.root / f"{key}.cache"
        try:
            if not path.is_file():
                return None
            if path.is_symlink() or path.stat().st_size > ENTRY_LIMIT:
                raise ValueError("invalid cache entry")
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if (len(members) > 16 or sum(member.file_size for member in members) > ENTRY_LIMIT
                        or any(member.compress_type != zipfile.ZIP_STORED for member in members)):
                    raise ValueError("invalid cache size")
                metadata = json.loads(archive.read("manifest.json"))
                records = metadata["variants"]
                names = ["manifest.json", *[f"{index}.jpg" for index in range(len(records))]]
                if metadata["key"] != key or archive.namelist() != names:
                    raise ValueError("invalid cache manifest")
                variants = [(record[0], archive.read(f"{index}.jpg"), record[1], record[2])
                            for index, record in enumerate(records)]
            with self.lock:
                os.utime(path, None)
                self.entries[path.name] = (path.stat().st_size, time.time())
            return variants
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError,
                zipfile.BadZipFile):
            self.discard(key)
            return None

    def save(self, key, variants):
        if self.root is None:
            return
        temporary = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            metadata = {"key": key, "variants": [[index, document, provenance]
                                                 for _, index, _, document, provenance, _ in variants]}
            with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("manifest.json", json.dumps(metadata, ensure_ascii=False))
                for index, (_, _, contents, _, _, _) in enumerate(variants):
                    archive.writestr(f"{index}.jpg", contents)
            size = temporary.stat().st_size
            if size <= min(ENTRY_LIMIT, self.max_bytes):
                with self.lock:
                    self.trim(size)
                    if sum(entry[0] for entry in self.entries.values()) + size <= self.max_bytes:
                        destination = self.root / f"{key}.cache"
                        temporary.replace(destination)
                        self.entries[destination.name] = (size, time.time())
        except (OSError, ValueError, zipfile.BadZipFile):
            with self.lock:
                self.errors += 1
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                with self.lock:
                    self.errors += 1

    def record(self, hit, count):
        with self.lock:
            if hit:
                self.hits += count
            else:
                self.misses += count

    def stats(self):
        with self.lock:
            return {"cache_hits": self.hits, "cache_misses": self.misses, "cache_errors": self.errors}


class ExportAcceleration:
    def __init__(self, workers=1, cache=None, checkpoint=None):
        self.workers = workers
        self.cache = cache
        self.external_checkpoint = checkpoint
        self.stop = threading.Event()
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="labelpilot-export-image")
        return self

    def __exit__(self, *exception):
        self.stop.set()
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def checkpoint(self):
        if self.external_checkpoint:
            self.external_checkpoint()
        if self.stop.is_set():
            raise Problem("导出准备已停止", 409)

    def map(self, function, values):
        if self.pool is None:
            for value in values:
                self.checkpoint()
                yield function(value)
            return
        pending = deque()
        iterator = iter(values)
        for _ in range(self.workers):
            self.checkpoint()
            try:
                value = next(iterator)
            except StopIteration:
                break
            pending.append(self.pool.submit(function, value))
        while pending:
            self.checkpoint()
            yield pending.popleft().result()
            self.checkpoint()
            try:
                value = next(iterator)
            except StopIteration:
                continue
            pending.append(self.pool.submit(function, value))
