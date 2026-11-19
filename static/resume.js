"use strict";

const workspaceResume = (() => {
  const delay = 1800;
  let saveTimer, positionTimer, prefetchTimer, saving, prefetched, edit = 0, conflict = null;

  function storage(key, value, scope=localStorage) {
    try {
      if (value === undefined) return JSON.parse(scope.getItem(key));
      if (value === null) scope.removeItem(key);
      else scope.setItem(key, JSON.stringify(value));
    } catch (_) { return null; }
  }

  const preference = "labelpilot-auto-draft";
  const positionKey = id => `labelpilot-position-${id}`;
  const draftKey = (id, item) => `labelpilot-draft-${id}-${item}`;

  function show(text, error=false) {
    comparisonMessage("autoSaveStatus", text, error);
  }

  function capture() {
    if (!state.project || !state.current || !state.image) return;
    const area = viewport();
    storage(positionKey(state.project.id), {item:state.current.item.id, filters:filters(), offset:state.offset,
      selected:state.selected, version:state.current.version, scale:state.scale,
      center:[(area.width / 2 - state.panX) / state.scale, (area.height / 2 - state.panY) / state.scale]});
  }

  function observe() {
    if (positionTimer) return;
    positionTimer = setTimeout(() => { positionTimer = null; capture(); }, 250);
  }

  function position() {
    return state.project ? storage(positionKey(state.project.id)) : null;
  }

  function restore(remembered) {
    if (!remembered?.filters || typeof remembered.item !== "string") return;
    const {batch, status, query} = remembered.filters;
    if (!batch || state.project.batches.includes(batch)) element("batchSelect").value = batch;
    if ([...element("statusSelect").options].some(option => option.value === status)) element("statusSelect").value = status;
    if (typeof query === "string") element("searchInput").value = query;
    if (Number.isInteger(remembered.offset) && remembered.offset >= 0) state.offset = remembered.offset;
  }

  function loaded(remembered) {
    conflict = null;
    clearTimeout(saveTimer);
    const current = state.current;
    const draft = storage(draftKey(state.project.id, current.item.id), undefined, sessionStorage);
    show("");
    if (draft?.document?.shapes && state.current.version === draft.version && current.image_sha256 === draft.image_sha256
        && current.source_fingerprint === draft.source_fingerprint) {
      checkpoint();
      current.document = draft.document;
      changed();
      show("已恢复这个窗口的本机草稿");
    } else if (draft) {
      conflict = current;
      show("本机草稿与服务器版本冲突，可下载草稿后核对", true);
    }
    element("downloadLocalDraft").hidden = !conflict;
    if (remembered?.item === current.item.id && remembered.version === current.version) {
      const selection = remembered.selected;
      if (Number.isInteger(selection) && selection >= 0 && selection < current.document.shapes.length) selectShape(selection);
      if (Number.isFinite(remembered.scale) && remembered.scale > 0 && remembered.center?.every(Number.isFinite)) {
        const area = viewport();
        state.scale = remembered.scale;
        state.panX = area.width / 2 - remembered.center[0] * state.scale;
        state.panY = area.height / 2 - remembered.center[1] * state.scale;
        draw();
      }
    }
    capture();
    prefetch();
  }

  function backup() {
    if (!state.current || conflict === state.current || !element("autoDraft").checked) return;
    const {item, version, image_sha256, source_fingerprint, document} = state.current;
    storage(draftKey(state.project.id, item.id), {version, image_sha256, source_fingerprint, document}, sessionStorage);
  }

  function edited() {
    ++edit;
    backup();
    schedule();
  }

  function schedule() {
    clearTimeout(saveTimer);
    if (element("autoDraft").checked && state.dirty && conflict !== state.current) {
      saveTimer = setTimeout(() => autoSave(), delay);
    }
  }

  function savedManually() {
    clearTimeout(saveTimer);
    if (conflict === state.current) return;
    storage(draftKey(state.project.id, state.current.item.id), null, sessionStorage);
    show("编辑进度已保存");
  }

  function autoSave() {
    if (!state.current || !state.dirty || !element("autoDraft").checked || conflict === state.current) return;
    if (saving || state.busy || state.drag || state.rectangle || state.polygon.length || state.pendingShape
        || efficiency.blocksReview() || document.querySelector("dialog[open]")) { schedule(); return; }
    const current = state.current, revision = edit;
    const key = draftKey(state.project.id, current.item.id);
    const payload = {expected_version:current.version, status:"draft", document:copy(current.document),
      image_sha256:current.image_sha256, source_fingerprint:current.source_fingerprint};
    show("正在自动保存草稿…");
    saving = api(itemUrl(), "PUT", payload).then(response => {
      if (state.current !== current) return;
      if (revision === edit) {
        Object.assign(current, response);
        setDirty(false);
        setStatus(response.status);
        storage(key, null, sessionStorage);
        if (!state.busy) refreshList().catch(() => {});
      } else {
        current.version = response.version;
        current.updated = response.updated;
        backup();
        schedule();
      }
      show("草稿已自动保存");
      capture();
    }).catch(error => {
      if (state.current === current) {
        show(error.message + "。编辑内容已留在本机草稿，可手动保存或下载。", true);
        conflict = current;
        element("downloadLocalDraft").hidden = false;
      }
    }).finally(() => { saving = null; });
  }

  function wait() { return saving || Promise.resolve(); }

  async function loadImage(url) {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  }

  async function imageFor(current, project=state.project.id) {
    if (prefetched?.id === current.item.id && prefetched.project === project) {
      const next = await prefetched.promise;
      if (next?.sha256 === current.image_sha256) return next.image;
    }
    return loadImage(`/api/projects/${project}/items/${current.item.id}/image`);
  }

  function prefetch() {
    clearTimeout(prefetchTimer);
    const project = state.project?.id;
    const active = state.current?.item.id;
    if (!project || !active) return;
    prefetchTimer = setTimeout(async () => {
      if (state.project?.id !== project || state.current?.item.id !== active) return;
      const index = state.items.findIndex(item => item.id === active);
      if (index < 0) return;
      try {
        let next = state.items[index + 1]?.id;
        if (!next && state.offset + state.items.length < state.filtered) {
          const query = new URLSearchParams({...filters(), offset:state.offset + state.items.length, limit:1});
          const page = await api(`/api/projects/${project}/items?${query}`);
          next = page.items[0]?.id;
        }
        if (state.project?.id !== project || state.current?.item.id !== active || !next ||
            prefetched?.project === project && prefetched.id === next) return;
        const url = `/api/projects/${project}/items/${next}`;
        prefetched = {project, id:next, promise:api(url).then(async row => ({
          sha256:row.image_sha256, image:await loadImage(url + "/image")
        })).catch(() => null)};
      } catch (_) { /* A normal image load can still retry. */ }
    }, 280);
  }

  function downloadDraft() {
    if (!state.current) return;
    const draft = storage(draftKey(state.project.id, state.current.item.id), undefined, sessionStorage);
    if (!draft) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(draft.document, null, 2)], {type:"application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.current.item.photo_id}-local-draft.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }

  function install() {
    const toggle = element("autoDraft");
    toggle.checked = storage(preference) !== false;
    listen("autoDraft", "change", () => {
      storage(preference, toggle.checked);
      if (state.dirty) backup();
      schedule();
      if (!toggle.checked) show("");
      if (state.current) canvas.focus({preventScroll:true});
    });
    listen("downloadLocalDraft", "click", downloadDraft);
    window.addEventListener("beforeunload", capture);
  }

  return {install, position, restore, loaded, edited, wait, imageFor, observe, prefetch, savedManually};
})();
