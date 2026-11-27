import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelpilot.server import create_app
from labelpilot.store import Problem, Store, digest


@pytest.fixture
def workspace(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (160, 100), "white").save(images / "panel.png")
    store = Store(tmp_path / "tool")
    source = store.register({"name": "原项目", "image_root": str(images), "classes": ["text", "unit"]})
    return store, source["id"], images


def detail(store, project_id):
    return store.detail(project_id, store.project(project_id).items[0]["id"])


def save(store, project_id, status="reviewed", *, empty=False):
    current = detail(store, project_id)
    document = copy.deepcopy(current["document"])
    document["shapes"] = [] if empty else [{"label": "text", "shape_type": "rectangle", "points": [[10, 15], [90, 50]],
                                           "description": "原始属性", "attributes": {"custom": True}}]
    return store.save(project_id, current["item"]["id"], {**current, "document": document,
                      "expected_version": current["version"], "status": status, "confirmed_empty": empty})


def target(store, images, *, name="新项目", classes=None, task="detect"):
    return store.register({"name": name, "image_root": str(images), "classes": classes or ["unit", "text"], "task_type": task})["id"]


def candidates(store, project_id):
    return store.management.candidates(project_id, detail(store, project_id))


def reuse_payload(store, project_id, candidate):
    current = detail(store, project_id)
    return {"reference": candidate["reference"], "expected_version": current["version"], "image_sha256": current["image_sha256"]}


def test_archive_restore_survives_restart_and_preserves_history(workspace):
    store, source, images = workspace
    original = digest(images / "panel.png")
    save(store, source)
    item_id = detail(store, source)["item"]["id"]
    store.management.archive(source)
    assert source not in store.projects
    restarted = Store(store.home)
    row = restarted.management.listing()["projects"][0]
    assert row["state"] == "archived" and row["reviewed"] == 1 and row["total"] == 1
    assert restarted.management.restore(source)["id"] == source
    assert detail(restarted, source)["status"] == "reviewed"
    assert [row["version"] for row in restarted.history(source, item_id)] == [1, 0]
    assert digest(images / "panel.png") == original
    assert source in Store(store.home).projects


def test_missing_source_directory_still_allows_archive_management(workspace):
    store, source, images = workspace
    store.management.archive(source)
    renamed = images.with_name("disconnected")
    images.rename(renamed)
    restarted = Store(store.home)
    assert restarted.management.listing()["projects"][0]["state"] == "archived"
    with pytest.raises(Problem):
        restarted.management.restore(source)
    assert restarted.management.states()[source]["state"] == "archived"
    restarted.management.delete(source, "原项目")
    assert (renamed / "panel.png").is_file()


def test_delete_requires_archive_and_exact_confirmation_keeps_files_and_library(workspace, tmp_path):
    store, source, images = workspace
    save(store, source)
    export = tmp_path / "exports" / "classes.txt"
    export.parent.mkdir()
    export.write_text("text", encoding="utf-8")
    registry = store.registry_path.read_bytes()
    with pytest.raises(Problem) as caught:
        store.management.delete(source, "原项目")
    assert caught.value.status == 409
    store.management.archive(source)
    with pytest.raises(Problem) as caught:
        store.management.delete(source, "wrong")
    assert caught.value.status == 422
    result = store.management.delete(source, "原项目")
    assert result["retained_annotations"] == 1
    with store.connect() as connection:
        for table in ("annotations", "revisions", "intake_receipts", "review_routes"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table} WHERE project=?", (source,)).fetchone()[0] == 0
    assert (images / "panel.png").is_file() and export.read_text(encoding="utf-8") == "text"
    store.registry_path.write_bytes(registry)
    restarted = Store(store.home)
    assert source not in restarted.projects and not restarted.management.listing()["projects"]
    with pytest.raises(Problem):
        restarted.management.restore(source)
    destination = target(restarted, images)
    assert candidates(restarted, destination)["candidates"][0]["source_deleted"] is True


def test_request_lease_prevents_archive(workspace):
    store, source, _ = workspace
    with store.management.access(source):
        with pytest.raises(Problem) as caught:
            store.management.archive(source)
        assert caught.value.status == 409
    store.management.archive(source)
    with pytest.raises(Problem) as caught:
        with store.management.access(source):
            pass
    assert caught.value.status == 404


def test_same_bytes_different_name_reuses_independent_draft(workspace, tmp_path):
    store, source, images = workspace
    save(store, source)
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    (renamed / "other-name.png").write_bytes((images / "panel.png").read_bytes())
    destination = target(store, renamed)
    candidate = candidates(store, destination)["candidates"][0]
    result = store.management.reuse(destination, detail(store, destination)["item"]["id"], reuse_payload(store, destination, candidate))
    assert result["status"] == "draft" and result["document"]["checked"] is False
    assert result["document"]["imagePath"] == "other-name.png"
    assert result["document"]["shapes"][0]["attributes"] == {"custom": True}
    metadata = result["document"]["labelpilot"]
    assert metadata["human_reviewed"] is False and metadata["annotation_reuse"]["project"] == source
    assert detail(store, source)["version"] == 1
    assert not candidates(store, destination)["candidates"]
    current = detail(store, destination)
    current["document"]["shapes"][0]["points"][0][0] = 20
    store.save(destination, current["item"]["id"], {**current, "expected_version": current["version"], "status": "draft"})
    assert detail(store, source)["document"]["shapes"][0]["points"][0][0] == 10
    assert detail(store, destination)["document"]["labelpilot"]["annotation_reuse"]["project"] == source


def test_same_name_different_bytes_is_not_reused(workspace, tmp_path):
    store, source, _ = workspace
    save(store, source)
    different = tmp_path / "different"
    different.mkdir()
    Image.new("RGB", (160, 100), "red").save(different / "panel.png")
    assert candidates(store, target(store, different))["candidates"] == []


@pytest.mark.parametrize("status", ["draft", "skipped", "predicted"])
def test_unreviewed_annotations_are_not_reusable(workspace, status):
    store, source, images = workspace
    if status == "predicted":
        current = detail(store, source)
        store.save(source, current["item"]["id"], {**current, "expected_version": 0, "status": status}, predicted=True)
    else:
        save(store, source, status)
    assert candidates(store, target(store, images))["candidates"] == []


def test_task_and_class_compatibility(workspace):
    store, source, images = workspace
    save(store, source)
    assert not candidates(store, target(store, images, task="ocr"))["candidates"]
    result = candidates(store, target(store, images, classes=["unit"]))
    assert not result["candidates"] and "text" in result["incompatible"][0]["reason"]


def test_renamed_source_labels_remain_correct_after_source_deletion(workspace):
    store, source, images = workspace
    save(store, source)
    config = next(row for row in store.configs if row["id"] == source)
    config["classes"] = ["renamed_text", "unit"]
    config["label_aliases"] = {"text": "renamed_text"}
    destination = target(store, images, classes=["renamed_text"])
    assert len(candidates(store, destination)["candidates"]) == 1
    store.management.archive(source)
    store.management.delete(source, "原项目")
    current = detail(store, destination)
    row = candidates(store, destination)["candidates"][0]
    result = store.management.reuse(destination, current["item"]["id"], reuse_payload(store, destination, row))
    assert result["document"]["shapes"][0]["label"] == "renamed_text"


def test_default_export_excludes_reused_draft(workspace):
    from labelpilot.exporting import preview_export

    store, source, images = workspace
    save(store, source)
    destination = target(store, images)
    candidate = candidates(store, destination)["candidates"][0]
    store.management.reuse(destination, detail(store, destination)["item"]["id"], reuse_payload(store, destination, candidate))
    with pytest.raises(Problem) as caught:
        preview_export(store, destination, {"format": "yolo"})
    assert caught.value.status == 422
    save(store, destination)
    assert preview_export(store, destination, {"format": "yolo"})["images"] == 1


def test_multiple_sources_are_exposed_and_stale_choice_is_rejected(workspace):
    store, source, images = workspace
    save(store, source)
    second = target(store, images, name="第二来源")
    save(store, second)
    destination = target(store, images)
    rows = candidates(store, destination)["candidates"]
    assert len(rows) == 2
    chosen = next(row for row in rows if row["project"] == source)
    payload = reuse_payload(store, destination, chosen)
    save(store, source)
    with pytest.raises(Problem) as caught:
        store.management.reuse(destination, detail(store, destination)["item"]["id"], payload)
    assert caught.value.status == 409
    assert detail(store, destination)["status"] == "unstarted"


def test_empty_negative_reuse_requires_new_review_confirmation(workspace):
    store, source, images = workspace
    save(store, source, empty=True)
    destination = target(store, images)
    candidate = candidates(store, destination)["candidates"][0]
    assert candidate["empty"] is True
    store.management.reuse(destination, detail(store, destination)["item"]["id"], reuse_payload(store, destination, candidate))
    current = detail(store, destination)
    assert current["status"] == "draft" and not current["document"]["labelpilot"]["negative_sample_confirmed"]
    with pytest.raises(Problem) as caught:
        store.save(destination, current["item"]["id"], {**current, "expected_version": current["version"], "status": "reviewed"})
    assert caught.value.status == 422
    store.save(destination, current["item"]["id"], {**current, "expected_version": current["version"], "status": "reviewed", "confirmed_empty": True})


@pytest.mark.parametrize("existing", ["draft", "reviewed", "predicted", "imported"])
def test_existing_target_annotations_are_never_overwritten(workspace, tmp_path, existing):
    store, source, images = workspace
    save(store, source)
    destination = target(store, images)
    current = detail(store, destination)
    payload = reuse_payload(store, destination, candidates(store, destination)["candidates"][0])
    if existing == "predicted":
        store.save(destination, current["item"]["id"], {**current, "expected_version": 0, "status": existing}, predicted=True)
    elif existing == "imported":
        native = tmp_path / "native"
        native.mkdir()
        (native / "panel.json").write_text(json.dumps(detail(store, source)["document"]), encoding="utf-8")
        store.project(destination).config["annotation_root"] = str(native)
        store.rescan(destination)
    else:
        save(store, destination, existing)
    assert not candidates(store, destination)["candidates"]
    with pytest.raises(Problem):
        store.management.reuse(destination, current["item"]["id"], payload)


def test_library_is_updated_and_removed_when_review_is_revoked(workspace):
    store, source, images = workspace
    save(store, source)
    destination = target(store, images)
    assert len(candidates(store, destination)["candidates"]) == 1
    save(store, source, "draft")
    assert not candidates(store, destination)["candidates"]
    save(store, source)
    assert candidates(store, destination)["candidates"][0]["version"] == 3


def test_legacy_reviewed_records_backfill_without_reading_source_images(workspace, monkeypatch):
    store, source, images = workspace
    save(store, source)
    with store.connect() as connection:
        connection.execute("DELETE FROM annotation_library")
        connection.execute("DELETE FROM project_management_meta")
    monkeypatch.setattr("labelpilot.store.digest", lambda *args, **kwargs: pytest.fail("startup must not hash source images"))
    restarted = Store(store.home)
    with restarted.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM annotation_library").fetchone()[0] == 1


def test_api_security_busy_guards_and_archived_access(workspace):
    store, source, _ = workspace
    app = create_app(store.home)
    headers = {"X-LabelPilot-Token": app.state.token}
    with TestClient(app) as client:
        endpoint = f"/api/project-management/{source}/archive"
        assert client.post(endpoint).status_code == 403
        assert client.post(endpoint, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        for manager in (app.state.exports, app.state.export_previews, app.state.automation):
            manager.records["busy-test"] = {"project": source, "status": "running"}
            assert client.post(endpoint, headers=headers).status_code == 409
            del manager.records["busy-test"]
        app.state.inference.lock.acquire()
        try:
            assert client.post(endpoint, headers=headers).status_code == 409
        finally:
            app.state.inference.lock.release()
        assert client.post(endpoint, headers=headers).status_code == 200
        assert client.get("/api/projects").json()["archived_count"] == 1
        assert client.get(f"/api/projects/{source}/items").status_code == 404
        assert client.get("/api/project-management").json()["projects"][0]["state"] == "archived"
        assert client.post(f"/api/project-management/{source}/restore", headers=headers).status_code == 200
        assert client.get(f"/api/projects/{source}/items").status_code == 200


def test_automation_does_not_poll_archived_projects(workspace, monkeypatch):
    store, source, _ = workspace
    app = create_app(store.home)
    automation = app.state.automation
    automation.settings[source] = {"intake": {"watch": True}}
    app.state.store.management.archive(source)
    monkeypatch.setattr(automation, "settings_for", lambda *_: pytest.fail("archived watcher must not execute"))
    automation.schedule()


def test_request_lease_wait_does_not_block_server_event_loop(workspace, monkeypatch):
    store, source, _ = workspace
    app = create_app(store.home)
    started = threading.Event()
    original = app.state.store.management.access

    @contextmanager
    def waiting(project_id):
        started.set()
        with original(project_id):
            yield

    monkeypatch.setattr(app.state.store.management, "access", waiting)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as workers:
        app.state.store.lock.acquire()
        try:
            project_request = workers.submit(client.get, f"/api/projects/{source}/items")
            assert started.wait(3)
            health = workers.submit(client.get, "/api/health")
            assert health.result(timeout=3).status_code == 200
        finally:
            app.state.store.lock.release()
        assert project_request.result(timeout=3).status_code == 200
