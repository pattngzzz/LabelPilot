import json
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path

from .store import Problem


DETAILS_FILE = "frames.sqlite3"
REVIEW_FILTERS = ("old_missed", "new_missed", "old_false_positives", "new_false_positives",
                  "newly_missed", "fixed_missed")


@contextmanager
def write_details(run, *, working_directory=None):
    with tempfile.TemporaryDirectory(dir=working_directory or run, prefix=".frames-") as staging:
        database = Path(staging) / DETAILS_FILE
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executescript("""
                CREATE TABLE frames (
                    position INTEGER PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    photo_id TEXT NOT NULL,
                    image TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    truth TEXT NOT NULL,
                    old_predictions TEXT NOT NULL,
                    new_predictions TEXT
                );
                CREATE TABLE issues (
                    label TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (label, kind, position)
                ) WITHOUT ROWID;
            """)
            yield connection
        database.replace(run / DETAILS_FILE)


def frame_issues(labels, old, new):
    # The index identifies the same frozen truth box in both model passes.
    old_matches = {prediction["truth_index"] for prediction in old}
    new_matches = {prediction["truth_index"] for prediction in new}
    issues = Counter()
    for index, (label, _) in enumerate(labels):
        old_miss, new_miss = index not in old_matches, index not in new_matches
        if old_miss:
            issues[label, "old_missed"] += 1
        if new_miss:
            issues[label, "new_missed"] += 1
        if old_miss != new_miss:
            issues[label, "newly_missed" if new_miss else "fixed_missed"] += 1
    for kind, predictions in (("old_false_positives", old), ("new_false_positives", new)):
        issues.update((prediction["label"], kind) for prediction in predictions if prediction["truth_index"] is None)
    return issues


def record_frame(connection, role, position, item, labels, size, predictions):
    serialized = json.dumps(predictions, ensure_ascii=False, separators=(",", ":"))
    if role == "old":
        truth = json.dumps([{"label": label, "box": box} for label, box in labels], ensure_ascii=False)
        connection.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                           (position, item["id"], item["photo_id"], item["image"], item["sha256"], *size, truth, serialized))
        return
    previous, = connection.execute("SELECT old_predictions FROM frames WHERE position = ?", (position,)).fetchone()
    connection.execute("UPDATE frames SET new_predictions = ? WHERE position = ?", (serialized, position))
    issues = frame_issues(labels, json.loads(previous), predictions)
    connection.executemany("INSERT INTO issues VALUES (?, ?, ?, ?)",
                           ((name, kind, position, count) for (name, kind), count in issues.items()))


def review_frames(database, label, kind, offset, limit):
    if kind not in REVIEW_FILTERS:
        raise Problem("请选择有效的逐图筛选条件", 422)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        total, count = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM issues WHERE label = ? AND kind = ?", (label, kind)
        ).fetchone()
        items = connection.execute("""
            SELECT frames.id, frames.photo_id, issues.count
            FROM issues JOIN frames USING (position)
            WHERE issues.label = ? AND issues.kind = ?
            ORDER BY issues.position LIMIT ? OFFSET ?
        """, (label, kind, limit, offset)).fetchall()
    return {"items": [dict(item) for item in items], "total": total, "count": count, "offset": offset, "limit": limit}


def review_frame(database, frame_id, *, image=False):
    columns = "photo_id, image, sha256" if image else "id, photo_id, width, height, truth, old_predictions, new_predictions"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(f"SELECT {columns} FROM frames WHERE id = ?", (frame_id,)).fetchone()
    if row is None:
        raise Problem("这张图片不在本次模型对比中", 404)
    if image:
        return dict(row)
    return {"id": row["id"], "photo_id": row["photo_id"], "width": row["width"], "height": row["height"],
            "truth": json.loads(row["truth"]), "old": json.loads(row["old_predictions"]), "new": json.loads(row["new_predictions"])}
