"use strict";

const projectManagement = (() => {
  let projects = [], deletion = null, choices = null;

  function render() {
    const filter = element("managedProjectFilter").value;
    const rows = projects.filter(project => !filter || project.state === filter);
    element("managedProjectList").innerHTML = rows.map(project => `<article class="managed-project">
      <div><strong>${escapeHtml(project.name)}</strong><span class="pill">${project.state === "archived" ? "已归档" : project.available ? "使用中" : "目录不可用"}</span>
      <p>${project.total == null ? "图片数量未知" : project.total + " 张图片"} · ${project.reviewed} 张已审核 · ${escapeHtml(project.task_type)}</p>
      <p class="muted managed-path">${escapeHtml(project.image_root)}</p></div>
      <div class="managed-actions">${project.state === "archived"
        ? `<button type="button" class="button" data-project-action="restore" data-project-id="${escapeHtml(project.id)}">复原</button><button type="button" class="button danger" data-project-action="delete" data-project-id="${escapeHtml(project.id)}">永久删除</button>`
        : `<button type="button" class="button" data-project-action="archive" data-project-id="${escapeHtml(project.id)}">归档</button>`}</div>
    </article>`).join("") || '<p class="empty-list">没有符合条件的项目</p>';
    element("managedProjectCount").textContent = `${projects.length} 个项目 · ${projects.filter(project => project.state === "archived").length} 个已归档`;
  }

  async function refresh() {
    projects = (await api("/api/project-management")).projects;
    render();
  }

  async function open(archived=false) {
    if (state.busy || !await ensureSaved()) return;
    if (state.pendingShape || state.rectangle || state.polygon.length) return toast("请先完成或取消当前绘制", true);
    clearInterval(jobTimer);
    element("managedProjectFilter").value = archived ? "archived" : "";
    element("projectManagementDialog").showModal();
    await refresh();
  }

  async function reloadWorkspace(preferred) {
    const response = await api("/api/projects");
    state.projects = response.projects;
    element("projectSelect").innerHTML = state.projects.map(project => `<option value="${escapeHtml(project.id)}">${escapeHtml(project.name)}</option>`).join("");
    const project = state.projects.find(row => row.id === preferred) || state.projects.find(row => row.id === state.project?.id) || state.projects[0];
    if (project && project.id === state.project?.id) {
      state.project = project;
    } else if (project) {
      await selectProject(project.id);
    } else {
      clearProject();
    }
    await refresh();
  }

  async function action(project, operation) {
    if (state.busy || !await ensureSaved()) return;
    if (operation === "delete") {
      deletion = project;
      element("deleteProjectName").textContent = project.name;
      element("deleteProjectConfirmation").value = "";
      element("confirmDeleteProject").disabled = true;
      element("deleteProjectDialog").showModal();
      return;
    }
    if (operation === "archive" && !window.confirm(`归档「${project.name}」？\n原图、标注和历史全部保留，可在项目管理中复原。自动流程在归档期间暂停。`)) return;
    await execute(project, operation);
  }

  async function execute(project, operation) {
    setBusy(true, operation === "restore" ? "正在复原项目…" : "正在更新项目…");
    try {
      const result = await api(`/api/project-management/${project.id}/${operation}`, "POST",
        operation === "delete" ? {confirmation:element("deleteProjectConfirmation").value} : {});
      if (operation === "delete") element("deleteProjectDialog").close();
      setBusy(false);
      await reloadWorkspace(operation === "restore" ? project.id : null);
      toast(operation === "archive" ? "项目已归档，可在项目管理中复原" : operation === "restore" ? "项目已复原" : `项目已永久删除，保留 ${result.retained_annotations} 张已审核标注供同图复用；原始文件未删除`);
      if (result.warning) toast(result.warning, true);
    } finally {
      setBusy(false);
    }
  }

  function eligible() {
    return state.current?.status === "unstarted" && state.current.version === 0 && !state.current.item.source && !state.dirty;
  }

  function updateSourceNote() {
    const origin = state.current?.document.labelpilot?.annotation_reuse;
    if (origin) element("sourceNote").textContent = `同图标注来源：${origin.project_name} · 独立副本 · ${state.current.status === "reviewed" ? "本项目已审核" : "待人工复核，请核对后按 E 审核"}`;
  }

  async function applyCandidate(reference) {
    if (!eligible()) throw new Error("当前图片已有修改，不能用历史标注覆盖");
    const current = state.current;
    const result = await api(itemUrl("/reuse"), "POST", {reference, expected_version:current.version, image_sha256:current.image_sha256});
    Object.assign(current, result);
    state.undo = [];
    state.redo = [];
    state.selected = -1;
    taskUi.itemChanged();
    setStatus(result.status);
    renderObjects();
    draw();
    element("reuseAnnotationButton").hidden = true;
    const origin = result.document.labelpilot.annotation_reuse;
    element("sourceNote").textContent = `已复用「${origin.project_name}」的已审核标注，作为独立草稿；请核对后按 E 审核${origin.empty ? "（空图需再次确认）" : ""}`;
    await refreshList();
  }

  async function loaded() {
    choices = null;
    element("reuseAnnotationButton").hidden = true;
    updateSourceNote();
    if (!eligible() || state.project.reuse_annotations === false) return;
    try {
      if (sessionStorage.getItem(`labelpilot-draft-${state.project.id}-${state.current.item.id}`)) return;
      const current = state.current;
      const response = current.reuse_candidates || await api(itemUrl("/reuse"));
      if (state.current !== current || !eligible()) return;
      if (response.candidates.length === 1 && !response.incompatible.length) {
        await applyCandidate(response.candidates[0].reference);
        return;
      }
      if (!response.candidates.length && !response.incompatible.length) return;
      choices = response;
      element("reuseAnnotationButton").hidden = false;
      element("reuseAnnotationButton").textContent = response.candidates.length ? `选择同图标注（${response.candidates.length}）` : "同图标注标签不兼容";
      element("sourceNote").textContent = "发现同图历史标注；有多个来源或标签不兼容，未自动套用";
    } catch (error) {
      toast(`同图标注复用：${error.message}`, true);
    }
  }

  function showChoices() {
    if (!choices || !eligible() || state.busy) return;
    element("reuseAnnotationList").innerHTML = choices.candidates.map(row => `<article class="managed-project"><div><strong>${escapeHtml(row.project_name)}</strong><p>${row.objects} 个对象 · ${row.empty ? "已确认空图" : "已审核"} · 版本 ${row.version}${row.source_deleted ? " · 来自保留复用库" : ""}</p><p class="muted managed-path">${escapeHtml(row.relative)}</p></div><button class="button" type="button" data-reuse-reference="${row.reference}">复用为草稿</button></article>`).join("")
      + choices.incompatible.map(row => `<article class="managed-project"><div><strong>${escapeHtml(row.project_name)}</strong><p class="muted">${escapeHtml(row.reason)}。导入所需标签后重新打开本图。</p></div></article>`).join("");
    element("reuseAnnotationDialog").showModal();
  }

  function install() {
    document.body.insertAdjacentHTML("beforeend", `
      <dialog id="projectManagementDialog" class="project-management-dialog"><form method="dialog" class="dialog-heading"><h2>项目管理</h2><button class="icon-button" aria-label="关闭">×</button></form>
        <div class="dialog-body"><p class="muted">归档隐藏项目并保留全部标注，可随时复原；同图按文件内容指纹匹配，不按文件名匹配。归档期间自动流程暂停，复原后沿用原设置。</p>
        <div class="managed-actions"><select id="managedProjectFilter" aria-label="项目状态"><option value="">全部项目</option><option value="active">未归档</option><option value="archived">已归档</option></select><button class="button" id="refreshManagedProjects" type="button">刷新列表</button><span class="muted" id="managedProjectCount"></span></div><div id="managedProjectList"></div></div></dialog>
      <dialog id="deleteProjectDialog"><form method="dialog" class="dialog-heading"><h2>永久删除项目</h2><button class="icon-button" aria-label="取消删除">×</button></form><div class="dialog-body"><p>将删除项目配置、项目内标注和修订历史，项目无法复原。</p><p>原始图片、外部标签、模型和导出文件不删除；已审核标注的独立复用库保留，未审核草稿不会保留。</p><label>输入完整项目名称确认：<strong id="deleteProjectName"></strong><input id="deleteProjectConfirmation" autocomplete="off"></label></div><div class="dialog-actions"><button class="button danger" id="confirmDeleteProject" type="button" disabled>永久删除</button></div></dialog>
      <dialog id="reuseAnnotationDialog"><form method="dialog" class="dialog-heading"><h2>选择同图已有标注</h2><button class="icon-button" aria-label="关闭">×</button></form><div class="dialog-body"><p class="muted">不会覆盖当前项目已有标注。复用后是独立草稿，须人工核对和审核；修改副本不会影响来源项目。</p><div id="reuseAnnotationList"></div></div></dialog>`);
    listen("projectManagementButton", "click", () => open());
    listen("refreshManagedProjects", "click", refresh);
    listen("managedProjectFilter", "change", render);
    listen("managedProjectList", "click", event => {
      const button = event.target.closest("[data-project-action]");
      const project = projects.find(row => row.id === button?.dataset.projectId);
      if (project) return action(project, button.dataset.projectAction);
    });
    listen("projectManagementDialog", "close", () => {
      if (state.project) {
        clearInterval(jobTimer);
        jobTimer = setInterval(() => pollJobs().catch(() => {}), 3000);
      }
    });
    listen("deleteProjectConfirmation", "input", () => {
      element("confirmDeleteProject").disabled = !deletion || element("deleteProjectConfirmation").value !== deletion.name;
    });
    listen("confirmDeleteProject", "click", () => deletion && !state.busy ? execute(deletion, "delete") : null);
    listen("deleteProjectDialog", "close", () => { deletion = null; });
    listen("reuseAnnotationButton", "click", showChoices);
    listen("reuseAnnotationList", "click", async event => {
      const reference = event.target.closest("[data-reuse-reference]")?.dataset.reuseReference;
      if (!reference || state.busy) return;
      setBusy(true, "正在复用同图标注…");
      try {
        await applyCandidate(reference);
        element("reuseAnnotationDialog").close();
      } finally { setBusy(false); }
    });
    for (const id of ["projectManagementDialog", "deleteProjectDialog", "reuseAnnotationDialog"]) {
      listen(id, "cancel", event => { if (state.busy) event.preventDefault(); });
    }
  }

  return {install, open, loaded, updateSourceNote};
})();
