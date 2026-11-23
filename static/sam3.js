"use strict";

const sam3Ui = (() => {
  let resultItemId = null;
  let resultLabel = "";
  let config = null;

  function fillLabels() {
    const select = element("sam3Label");
    const prior = select.value;
    select.replaceChildren(...(state.project?.classes || []).map(item => new Option(item.name, item.name)));
    if ((state.project?.classes || []).some(item => item.name === prior)) select.value = prior;
    else if (state.project?.classes?.[state.activeClass]) select.value = state.project.classes[state.activeClass].name;
    if (!element("sam3Prompt").dataset.edited && select.value) element("sam3Prompt").value = select.value;
  }

  async function refresh() {
    config = await api("/api/sam3");
    element("sam3PythonPath").value = config.python_executable || "";
    element("sam3CheckpointPath").value = config.checkpoint_path || "";
    element("sam3Device").value = config.device || "cuda";
    element("sam3Status").textContent = config.message;
    element("sam3Release").disabled = !config.running;
    element("sam3Generate").disabled = !state.current || !config.configured || !config.installed;
    return config;
  }

  async function open() {
    if (!state.project || state.project.task_type !== "segment") return;
    fillLabels();
    renderCandidates();
    element("sam3Dialog").showModal();
    try { await refresh(); }
    catch (error) { element("sam3Status").textContent = error.message; }
  }

  async function browse(mode, field) {
    const selected = await api("/api/sam3/select", "POST", {mode, path:element(field).value});
    if (!selected.cancelled) element(field).value = selected.path;
  }

  async function saveConfig() {
    const python = element("sam3PythonPath").value.trim();
    const checkpoint = element("sam3CheckpointPath").value.trim();
    if (!python || !checkpoint) throw new Error("请先选择 SAM3 独立 Python 环境和本地权重文件");
    config = await api("/api/sam3/config", "POST", {
      python_executable:python, checkpoint_path:checkpoint, device:element("sam3Device").value,
    });
    element("sam3Status").textContent = config.message;
    return config;
  }

  function renderCandidates() {
    const list = element("sam3CandidateList");
    list.replaceChildren();
    const candidates = state.sam3Candidates || [];
    if (!candidates.some(Boolean)) {
      const empty = document.createElement("p");
      empty.className = "empty-list";
      empty.textContent = "尚无候选；选择对象描述后生成。";
      list.append(empty);
      return;
    }
    candidates.forEach((candidate, index) => {
      if (!candidate) return;
      const row = document.createElement("div");
      row.className = "sam3-candidate";
      const select = document.createElement("button");
      select.type = "button";
      select.className = "button";
      select.textContent = `候选 ${index + 1} · ${Math.round(candidate.score * 100)}%`;
      select.addEventListener("click", () => {
        state.sam3PreviewIndex = index;
        draw();
      });
      const apply = document.createElement("button");
      apply.type = "button";
      apply.className = "button primary";
      apply.textContent = "采用此轮廓";
      apply.addEventListener("click", () => {
        try { applyCandidate(index); } catch (error) { element("sam3Status").textContent = error.message; }
      });
      row.append(select, apply);
      list.append(row);
    });
    const applyAll = document.createElement("button");
    applyAll.type = "button";
    applyAll.className = "button";
    applyAll.textContent = "采用全部候选";
    applyAll.addEventListener("click", () => {
      try { applyAllCandidates(); } catch (error) { element("sam3Status").textContent = error.message; }
    });
    list.append(applyAll);
  }

  async function generate() {
    if (!state.current || state.busy || !state.project || state.project.task_type !== "segment") return;
    const label = element("sam3Label").value;
    const prompt = element("sam3Prompt").value.trim();
    if (!label || !prompt) throw new Error("请选择项目标签并输入对象描述");
    const itemId = state.current.item.id;
    resultItemId = itemId;
    resultLabel = label;
    state.sam3Candidates = [];
    state.sam3PreviewIndex = -1;
    renderCandidates();
    element("sam3Generate").disabled = true;
    element("sam3Status").textContent = "SAM3 正在加载模型并分析当前图片；期间 YOLO 推理和训练会暂停…";
    draw();
    try {
      const result = await api(itemUrl("/sam3"), "POST", {
        prompt, label, threshold:Number(element("sam3Threshold").value),
      });
      if (!state.current || state.current.item.id !== itemId) {
        element("sam3Status").textContent = "推理完成，但当前图片已切换；请回到原图重新生成。";
        return;
      }
      resultLabel = result.label;
      state.sam3Candidates = result.candidates || [];
      state.sam3PreviewIndex = state.sam3Candidates.length ? 0 : -1;
      element("sam3Status").textContent = `SAM3 返回 ${state.sam3Candidates.length} 个候选。请选择轮廓并人工核对后再应用。`;
      renderCandidates();
      draw();
    } finally {
      element("sam3Generate").disabled = false;
      await refresh().catch(() => {});
    }
  }

  function applyCandidate(index) {
    if (!state.current || state.current.item.id !== resultItemId) throw new Error("图片已切换，请重新生成 SAM3 候选");
    const candidate = state.sam3Candidates[index];
    if (!candidate || !Array.isArray(candidate.points) || candidate.points.length < 3) return;
    const classIndex = state.project.classes.findIndex(item => item.name === resultLabel);
    if (classIndex < 0) throw new Error("候选标签已不存在，请重新选择");
    state.activeClass = classIndex;
    renderClasses();
    addShape(copy(candidate.points), "polygon", {label:resultLabel, score:candidate.score});
    recordApplication(1);
    state.sam3Candidates[index] = null;
    state.sam3PreviewIndex = -1;
    renderCandidates();
    element("sam3Status").textContent = "候选已加入当前标注草稿；仍需检查并保存 / 审核。";
  }

  function applyAllCandidates() {
    if (!state.current || state.current.item.id !== resultItemId) throw new Error("图片已切换，请重新生成 SAM3 候选");
    const candidates = state.sam3Candidates.filter(candidate => candidate?.points?.length >= 3);
    const classIndex = state.project.classes.findIndex(item => item.name === resultLabel);
    if (!candidates.length || classIndex < 0) return;
    state.activeClass = classIndex;
    checkpoint();
    for (const candidate of candidates) {
      state.current.document.shapes.push({label:resultLabel,points:copy(candidate.points),shape_type:"polygon",
        score:candidate.score,group_id:null,description:"",flags:{},attributes:{},difficult:false});
    }
    recordApplication(candidates.length);
    state.selected = state.current.document.shapes.length - 1;
    state.sam3Candidates = [];
    state.sam3PreviewIndex = -1;
    renderClasses();
    changed();
    renderCandidates();
    element("sam3Status").textContent = `已加入 ${candidates.length} 个候选轮廓到草稿；仍需检查并保存 / 审核。`;
  }

  function recordApplication(count) {
    const metadata = state.current.document.labelpilot || {};
    const history = Array.isArray(metadata.sam3_suggestions) ? [...metadata.sam3_suggestions] : [];
    history.push({prompt:element("sam3Prompt").value.trim(), label:resultLabel, applied_count:count,
      applied_at:new Date().toISOString()});
    state.current.document.labelpilot = {...metadata, sam3_suggestions:history.slice(-50)};
  }

  async function release() {
    element("sam3Release").disabled = true;
    try {
      const result = await api("/api/sam3/release", "POST", {});
      element("sam3Status").textContent = result.message;
    } finally {
      await refresh().catch(() => {});
    }
  }

  element("sam3Button").addEventListener("click", () => open().catch(error => toast(error.message, true)));
  element("sam3BrowsePython").addEventListener("click", () => browse("sam3_python", "sam3PythonPath").catch(error => toast(error.message, true)));
  element("sam3BrowseCheckpoint").addEventListener("click", () => browse("sam3_checkpoint", "sam3CheckpointPath").catch(error => toast(error.message, true)));
  element("sam3SaveConfig").addEventListener("click", () => saveConfig().catch(error => toast(error.message, true)));
  element("sam3Generate").addEventListener("click", () => generate().catch(error => {
    element("sam3Status").textContent = error.message;
    toast(error.message, true);
  }));
  element("sam3Release").addEventListener("click", () => release().catch(error => toast(error.message, true)));
  element("sam3Prompt").addEventListener("input", event => { event.target.dataset.edited = "true"; });
  element("sam3Label").addEventListener("change", event => {
    if (!element("sam3Prompt").dataset.edited) element("sam3Prompt").value = event.target.value;
  });
  element("sam3Threshold").addEventListener("input", event => {
    element("sam3ThresholdValue").textContent = Number(event.target.value).toFixed(2);
  });
  element("sam3Dialog").addEventListener("close", () => {
    state.sam3PreviewIndex = -1;
    draw();
  });
  document.addEventListener("labelpilot-item-changed", renderCandidates);

  return {open, refresh, renderCandidates};
})();
