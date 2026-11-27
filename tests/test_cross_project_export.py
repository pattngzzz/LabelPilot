import copy
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from labelpilot.export_jobs import ExportJobs
from labelpilot.exporting import preview_export
from labelpilot.store import Problem, Store, digest


def save_ocr(store, project_id, name, *, text="25 C", label="text", status="reviewed"):
    item = next(row for row in store.project(project_id).items if row["photo_id"] == name)
    current = store.detail(project_id, item["id"])
    document = copy.deepcopy(current["document"])
    document["shapes"] = [{"label": label, "shape_type": "polygon", "points": [[10, 10], [80, 10], [80, 35], [10, 35]],
                           "text_raw": text, "readability": "readable", "value_role": "unknown"}]
    store.save(project_id, item["id"], {**current, "document": document,
               "expected_version": current["version"], "status": status})
    return store.detail(project_id, item["id"])


def run_export(store, jobs, project_id, options):
    preview = preview_export(store, project_id, options)
    record = jobs.start(project_id, {**options, "preview_token": preview["preview_token"]})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = jobs.get(project_id, record["id"])
        if record["status"] in ("completed", "failed", "cancelled"):
            jobs.worker.join(timeout=2)
            assert record["status"] == "completed", record
            return record["result"], preview
        time.sleep(0.01)
    pytest.fail("Cross-project export did not finish")


def dataset_hashes(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def cross_projects(tmp_path):
    store = Store(tmp_path / "tool")
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
    for index, color in enumerate(("white", "red", "blue")):
        Image.new("RGB", (120, 60), color).save(roots[0] / f"panel-{index}.png")
    (roots[1] / "panel-0.png").write_bytes((roots[0] / "panel-0.png").read_bytes())
    Image.new("RGB", (120, 60), "green").save(roots[1] / "panel-1.png")
    (roots[1] / "zz-copy.png").write_bytes((roots[1] / "panel-1.png").read_bytes())
    Image.new("RGB", (120, 60), "black").save(roots[1] / "draft.png")
    first = store.register({"name": "Source A", "task_type": "ocr", "image_root": str(roots[0]), "classes": ["text"]})["id"]
    second = store.register({"name": "Source B", "task_type": "ocr", "image_root": str(roots[1]), "classes": ["reading", "text"]})["id"]
    for name in ("panel-0", "panel-1", "panel-2"):
        save_ocr(store, first, name)
    save_ocr(store, second, "panel-0")
    for name in ("panel-1", "zz-copy"):
        save_ocr(store, second, name, label="reading")
    save_ocr(store, second, "draft", status="draft")
    jobs = ExportJobs(store)
    try:
        yield store, jobs, first, second
    finally:
        jobs.shutdown()


def initial_export(store, jobs, project_id, augmentation=None):
    options = {"format": "paddleocr", "destination": str(store.home),
               "split": {"train": 34, "val": 33, "test": 33, "unit": "image", "seed": 42}}
    if augmentation:
        options["augmentation"] = {"preset": augmentation, "copies": 1, "seed": 42}
    result, _ = run_export(store, jobs, project_id, options)
    root = Path(result["directory"])
    manifest = json.loads((root / "export_manifest.json").read_bytes())
    return root, manifest


@pytest.mark.parametrize("augmentation", [None, "ocr_gentle", "all"])
def test_cross_project_append_deduplicates_and_preserves_sources_and_settings(cross_projects, augmentation):
    store, jobs, first, second = cross_projects
    root, original = initial_export(store, jobs, first, augmentation)
    before = dataset_hashes(root)
    revisions = {project_id: store.overlay(project_id) for project_id in (first, second)}
    options = {"format": "paddleocr", "destination": str(root), "incremental": True, "merge_projects": True}
    result, preview = run_export(store, jobs, second, options)
    change = preview["incremental"]
    assert (change["added"], change["updated"], change["skipped"], change["deduplicated"]) == (1, 0, 2, 2)
    assert change["cross_project"] and change["settings_preserved"]
    manifest = json.loads((root / "export_manifest.json").read_bytes())
    assert manifest["project"] == first
    assert {source["id"] for source in manifest["source_projects"]} == {first, second}
    assert manifest["classes"] == ["text", "reading"]
    assert manifest["augmentation"] == original["augmentation"]
    assert manifest["split_config"] == original["split_config"]
    originals = [entry for entry in manifest["items"] if "source_id" not in entry]
    assert manifest["original_images"] == len(originals) == len({entry["sha256"] for entry in originals}) == 4
    assert len({entry["id"] for entry in manifest["items"]}) == len(manifest["items"])
    for entry in original["items"]:
        retained = next(row for row in manifest["items"] if row["id"] == entry["id"])
        assert retained["image"] == entry["image"] and retained["split"] == entry["split"]
    shared = next(entry for entry in originals if entry["photo_id"] == "panel-0")
    assert {source["project"] for source in shared["sources"]} == {first, second}
    added = next(entry for entry in originals if entry["source_project"] == second)
    assert len(added["sources"]) == 2 and added["split"] == "train"
    assert result["integrity_check"]["passed"]
    for name, fingerprint in before.items():
        if name.startswith(("images/", "annotations/", "recognition/")):
            assert digest(root / name) == fingerprint
    for project_id in (first, second):
        assert store.overlay(project_id) == revisions[project_id]
    repeated, preview = run_export(store, jobs, second, options)
    assert preview["images"] == repeated["incremental"]["added"] == repeated["incremental"]["updated"] == 0
    assert repeated["incremental"]["skipped"] == 3
    assert json.loads((root / "export_manifest.json").read_bytes())["original_images"] == 4
    run_export(store, jobs, first, options)
    manifest = json.loads((root / "export_manifest.json").read_bytes())
    for subset, counts in manifest["ocr"]["splits"].items():
        assert len((root / f"det/{subset}.txt").read_text(encoding="utf-8").splitlines()) == counts["images"]
        assert len((root / f"rec/{subset}.txt").read_text(encoding="utf-8").splitlines()) == counts["recognition_crops"]


@pytest.mark.parametrize("conflict", ["text", "geometry", "attributes", "pending"])
def test_cross_project_conflicting_annotations_do_not_modify_target(cross_projects, conflict):
    store, jobs, first, second = cross_projects
    root, _ = initial_export(store, jobs, first)
    name = "zz-copy" if conflict == "pending" else "panel-0"
    current = save_ocr(store, second, name, text="different" if conflict in ("text", "pending") else "25 C",
                       label="reading" if conflict == "pending" else "text")
    if conflict in ("geometry", "attributes"):
        document = copy.deepcopy(current["document"])
        if conflict == "geometry":
            document["shapes"][0]["points"][0][0] += 1
        else:
            document["shapes"][0]["attributes"] = {"unit_verified": False}
        store.save(second, current["item"]["id"], {**current, "document": document,
                   "expected_version": current["version"], "status": "reviewed"})
    before = dataset_hashes(root)
    with pytest.raises(Problem, match="同图标注冲突"):
        preview_export(store, second, {"format": "paddleocr", "destination": str(root),
                       "incremental": True, "merge_projects": True})
    assert dataset_hashes(root) == before


@pytest.mark.parametrize("extra", [{"merge_projects": False}, {"merge_projects": "yes"}, {"reviewed_only": False}])
def test_cross_project_requires_explicit_opt_in_and_reviewed_data(cross_projects, extra):
    store, jobs, first, second = cross_projects
    root, _ = initial_export(store, jobs, first)
    before = dataset_hashes(root)
    with pytest.raises(Problem):
        preview_export(store, second, {"format": "paddleocr", "destination": str(root),
                       "incremental": True, "merge_projects": True, **extra})
    assert dataset_hashes(root) == before


@pytest.mark.parametrize("unit", [None, "image", "directory"])
def test_cross_project_inherits_target_split_mode_and_legacy_sources(cross_projects, unit):
    store, jobs, first, second = cross_projects
    options = {"format": "paddleocr", "destination": str(store.home)}
    if unit:
        options["split"] = {"train": 100, "val": 0, "test": 0, "seed": 7, "unit": unit}
    result, _ = run_export(store, jobs, first, options)
    root = Path(result["directory"])
    manifest_path = root / "export_manifest.json"
    original = json.loads(manifest_path.read_bytes())
    for entry in original["items"]:
        entry.pop("source_project", None)
        entry.pop("source_item", None)
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    result, _ = run_export(store, jobs, second, {"format": "paddleocr", "destination": str(root),
                          "incremental": True, "merge_projects": True})
    merged = json.loads(manifest_path.read_bytes())
    assert merged["split_config"] == original["split_config"]
    assert merged["original_images"] == 4
    assert {entry["split"] for entry in merged["items"]} == {"train" if unit else "unassigned"}
    assert all(entry.get("sources") for entry in merged["items"])


def test_cross_project_original_source_can_update_an_unshared_sample(cross_projects):
    store, jobs, first, second = cross_projects
    save_ocr(store, second, "zz-copy", label="reading", status="draft")
    root, _ = initial_export(store, jobs, first, "all")
    options = {"format": "paddleocr", "destination": str(root), "incremental": True, "merge_projects": True}
    run_export(store, jobs, second, options)
    before = json.loads((root / "export_manifest.json").read_bytes())
    previous = next(entry for entry in before["items"] if entry["source_project"] == second and "source_id" not in entry)
    save_ocr(store, second, "panel-1", label="reading", text="26 C")
    result, preview = run_export(store, jobs, second, options)
    assert preview["incremental"]["updated"] == 1 and result["incremental"]["added"] == 0
    merged = json.loads((root / "export_manifest.json").read_bytes())
    updated = next(entry for entry in merged["items"] if entry["id"] == previous["id"])
    assert updated["image"] == previous["image"] and updated["split"] == previous["split"]
    assert updated["revision"] == previous["revision"] + 1 and len(updated["sources"]) == 1
    assert merged["original_images"] == 4
    assert merged["augmented_images"] == before["augmented_images"]


def test_cross_project_shared_sample_cannot_be_silently_overwritten(cross_projects):
    store, jobs, first, second = cross_projects
    root, _ = initial_export(store, jobs, first)
    options = {"format": "paddleocr", "destination": str(root), "incremental": True, "merge_projects": True}
    run_export(store, jobs, second, options)
    save_ocr(store, first, "panel-0", text="changed by A")
    before = dataset_hashes(root)
    with pytest.raises(Problem, match="同图标注冲突"):
        preview_export(store, first, options)
    assert dataset_hashes(root) == before


def test_cross_project_publish_failure_rolls_back_target(cross_projects, monkeypatch):
    store, jobs, first, second = cross_projects
    root, _ = initial_export(store, jobs, first)
    options = {"format": "paddleocr", "destination": str(root), "incremental": True, "merge_projects": True}
    preview = preview_export(store, second, options)
    before = dataset_hashes(root)
    original = Path.replace
    def fail_manifest(path, target):
        if path.name == "export_manifest.json" and path.parent.name.endswith(".part"):
            raise OSError("simulated cross-project publish failure")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", fail_manifest)
    record = jobs.start(second, {**options, "preview_token": preview["preview_token"]})
    jobs.worker.join(timeout=15)
    assert not jobs.worker.is_alive()
    record = jobs.get(second, record["id"])
    assert record["status"] == "failed" and "simulated" in record["error"]
    assert dataset_hashes(root) == before


def test_cross_project_update_cannot_absorb_an_outdated_duplicate(cross_projects):
    store, jobs, first, second = cross_projects
    save_ocr(store, second, "zz-copy", label="reading", status="draft")
    root, _ = initial_export(store, jobs, first)
    options = {"format": "paddleocr", "destination": str(root), "incremental": True, "merge_projects": True}
    run_export(store, jobs, second, options)
    save_ocr(store, second, "panel-1", label="reading", text="26 C")
    save_ocr(store, second, "zz-copy", label="reading", text="25 C")
    before = dataset_hashes(root)
    with pytest.raises(Problem, match="同图标注冲突"):
        preview_export(store, second, options)
    assert dataset_hashes(root) == before


@pytest.mark.parametrize("damage", ["retained_image", "annotation", "source_reference"])
def test_cross_project_rejects_damaged_target_without_writes(cross_projects, damage):
    store, jobs, first, second = cross_projects
    root, manifest = initial_export(store, jobs, first)
    entry = next(row for row in manifest["items"] if row["photo_id"] == ("panel-2" if damage == "retained_image" else "panel-0"))
    if damage == "retained_image":
        (root / entry["image"]).unlink()
    elif damage == "annotation":
        annotation = next(name for name in entry["files"] if name.startswith("annotations/"))
        (root / annotation).write_text("{}", encoding="utf-8")
    else:
        entry["sources"] = [{"project": first, "item": entry["id"], "revision": entry["revision"]}] * 2
        (root / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = dataset_hashes(root)
    with pytest.raises(Problem):
        preview_export(store, second, {"format": "paddleocr", "destination": str(root),
                       "incremental": True, "merge_projects": True})
    assert dataset_hashes(root) == before


def test_cross_project_does_not_bypass_other_format_restrictions(cross_projects):
    store, jobs, first, second = cross_projects
    result, _ = run_export(store, jobs, first, {"format": "labelme", "destination": str(store.home)})
    root = Path(result["directory"])
    with pytest.raises(Problem, match="PaddleOCR"):
        preview_export(store, second, {"format": "labelme", "destination": str(root),
                       "incremental": True, "merge_projects": True})
