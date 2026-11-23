"use strict";

const sam3Ui = (() => {
  let resultItemId = null;
  let resultLabel = "";
  let resultPrompt = "";
  let resultBackend = "";
  let generating = false;
  let resultVersion = null;
  let resultImageHash = null;
  let resultProjectId = null;
  let config = null;

  function fillLabels() {
    const select = element("sam3Label");
    const prior = select.value;
    select.replaceChildren(...(state.project?.classes || []).map(item => new Option(item.name, item.name)));
    if ((state.project?.classes || []).some(item => item.name === prior)) select.value = prior;
    else if (state.project?.classes?.[state.activeClass]) select.value = state.project.classes[state.activeClass].name;
    if (!element("sam3Prompt").dataset.edited && select.value) element("sam3Prompt").value = select.value.replace(/_/g, " ");
  }

  async function refresh() {
    config = await api("/api/sam3");
    element("sam3PythonPath").value = config.python_executable || "";
    element("sam3CheckpointPath").value = config.checkpoint_path || "";
    element("sam3Device").value = config.device || "cuda";
    element("sam3Backend").value = config.backend || "pytorch";
    element("sam3Status").textContent = config.message;
    element("sam3Release").disabled = !config.running;
    element("sam3Generate").disabled = !state.current || !config.configured || !config.installed;
    element("sam3Batch").disabled = !config.configured || !config.installed;
    return config;
  }

  async function open() {
    if (!state.project || !["detect", "segment"].includes(state.project.task_type)) return;
    fillLabels();
    renderCandidates();
    element("sam3Dialog").returnValue = "";
    element("sam3Dialog").showModal();
    renderPreview();
    try { await refresh(); }
    catch (error) { element("sam3Status").textContent = error.message; }
  }

  async function browse(mode, field) {
    const selected = await api("/api/sam3/select", "POST", {mode, path:element(field).value});
    if (!selected.cancelled) {
      element(field).value = selected.path;
      configDirty();
    }
  }

  function configDirty() {
    element("sam3Generate").disabled = true;
    element("sam3Batch").disabled = true;
    element("sam3Status").textContent = "配置尚未保存，请先保存配置再生成候选。";
  }

  async function saveConfig() {
    const python = element("sam3PythonPath").value.trim();
    const checkpoint = element("sam3CheckpointPath").value.trim();
    if (!python || !checkpoint) throw new Error("请先选择 SAM3 独立 Python 环境和本地权重文件");
    config = await api("/api/sam3/config", "POST", {
      python_executable:python, checkpoint_path:checkpoint, device:element("sam3Device").value,
      backend:element("sam3Backend").value,
    });
    await refresh();
    return config;
  }

  function renderCandidates() {
    renderPreview();
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
      select.textContent = `预览 · ${candidate.label || resultLabel} · 候选 ${index + 1} · ${Math.round(candidate.score * 100)}%${candidate.source_index == null ? "" : " · 对应框 " + (candidate.source_index + 1)}`;
      select.addEventListener("click", () => {
        state.sam3PreviewIndex = index;
        element("sam3Dialog").close("preview");
        renderPreview();
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

  function renderPreview() {
    const candidate = state.sam3Candidates[state.sam3PreviewIndex];
    element("sam3PreviewBar").hidden = !candidate || element("sam3Dialog").open;
    element("sam3PreviewDescription").textContent = candidate ? `SAM3 候选 · ${candidate.label || resultLabel} · ${Math.round(candidate.score * 100)}% · 请检查轮廓后采用` : "";
  }

  async function generate() {
    if (generating || !state.current || state.busy || !state.project || !["detect", "segment"].includes(state.project.task_type)) return;
    const boxes = element("sam3Mode").value === "boxes";
    if (boxes && !await ensureSaved()) return;
    const label = element("sam3Label").value;
    const prompt = element("sam3Prompt").value.trim();
    if (!boxes && (!label || !prompt)) throw new Error("请选择项目标签并输入对象描述");
    const itemId = state.current.item.id;
    resultItemId = itemId;
    resultLabel = label;
    resultPrompt = prompt;
    resultBackend = config?.backend || "pytorch";
    resultProjectId = state.project.id;
    resultVersion = state.current.version;
    resultImageHash = state.current.image_sha256;
    generating = true;
    state.sam3Candidates = [];
    state.sam3PreviewIndex = -1;
    renderCandidates();
    element("sam3Generate").disabled = true;
    element("sam3Status").textContent = "SAM3 正在加载模型并分析当前图片；期间 YOLO 推理和训练会暂停…";
    draw();
    try {
      const result = await api(boxes ? itemUrl("/contours") : itemUrl("/sam3"), "POST", boxes ? boxOptions(false) : {
        prompt, label, threshold:Number(element("sam3Threshold").value),
      });
      if (!state.current || state.current.item.id !== itemId || state.project.id !== resultProjectId) {
        element("sam3Status").textContent = "推理完成，但当前图片已切换；请回到原图重新生成。";
        return;
      }
      resultLabel = result.label || label;
      state.sam3Candidates = result.candidates || [];
      state.sam3PreviewIndex = state.sam3Candidates.length ? 0 : -1;
      element("sam3Status").textContent = `SAM3 返回 ${state.sam3Candidates.length} 个候选${result.unmatched ? "，另有 " + result.unmatched + " 个框未找到可靠轮廓（原框保留）" : ""}。请核对后采用；关联轮廓会替换对应框。`;
      renderCandidates();
      draw();
    } finally {
      generating = false;
      element("sam3Generate").disabled = false;
      await refresh().catch(() => {});
    }
  }

  function applyCandidate(index) {
    checkResult();
    const candidate = state.sam3Candidates[index];
    if (!candidate || !Array.isArray(candidate.points) || candidate.points.length < 3) return;
    const label = candidate.label || resultLabel;
    const classIndex = state.project.classes.findIndex(item => item.name === label);
    if (classIndex < 0) throw new Error("候选标签已不存在，请重新选择");
    state.activeClass = classIndex;
    renderClasses();
    checkSource(candidate);
    checkpoint();
    applyShape(candidate);
    recordApplication(1);
    state.sam3Candidates[index] = null;
    state.sam3PreviewIndex = -1;
    renderCandidates();
    changed();
    element("sam3Status").textContent = "候选已加入当前标注草稿；仍需检查并保存 / 审核。";
  }

  function applyAllCandidates() {
    checkResult();
    const candidates = state.sam3Candidates.filter(candidate => candidate?.points?.length >= 3);
    if (!candidates.length) return;
    candidates.forEach(checkSource);
    checkpoint();
    for (const candidate of candidates) {
      applyShape(candidate);
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
    history.push({prompt:resultPrompt, backend:resultBackend, label:resultLabel, applied_count:count,
      applied_at:new Date().toISOString()});
    state.current.document.labelpilot = {...metadata, sam3_suggestions:history.slice(-50)};
  }

  function checkResult() {
    if (!state.current || state.current.item.id !== resultItemId || state.project.id !== resultProjectId) throw new Error("图片已切换，请重新读取候选");
    if (state.current.image_sha256 !== resultImageHash) throw new Error("图片已变化，请重新生成 / 读取候选");
  }

  function canonical(value) {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
    return value;
  }

  function checkSource(candidate) {
    const label = candidate.label || resultLabel;
    if (!state.project.classes.some(item => item.name === label)) throw new Error("候选标签已不存在");
    if (candidate.source_index != null && JSON.stringify(canonical(state.current.document.shapes[candidate.source_index])) !== JSON.stringify(canonical(candidate.source_shape))) {
      throw new Error("对应检测框已被修改或删除，请重新生成轮廓；未覆盖现有标注");
    }
  }

  function applyShape(candidate) {
    const source = candidate.source_shape;
    const shape = {...(source ? copy(source) : {group_id:null,description:"",flags:{},attributes:{},difficult:false}),
      label:candidate.label || resultLabel, points:copy(candidate.points), shape_type:"polygon", score:candidate.score,
      prediction_origin:"sam3_prediction", prediction_id:crypto.randomUUID(),
      sam3:{prompt:candidate.prompt || resultPrompt,backend:resultBackend,created_at:new Date().toISOString()}};
    if (source) {
      shape.contour_source = {shape:copy(source), source_index:candidate.source_index, source_key:candidate.source_key,
        image_sha256:resultImageHash, base_version:resultVersion, box_iou:candidate.box_iou};
      state.current.document.shapes[candidate.source_index] = shape;
      state.selected = candidate.source_index;
    } else {
      state.current.document.shapes.push(shape);
      state.selected = state.current.document.shapes.length - 1;
    }
  }

  function boxOptions(batch) {
    const scope = element("sam3BoxScope").value;
    if (batch && scope === "selected") throw new Error("批量任务请选择所有类别或当前标签，不能使用单张选中框");
    if (scope === "selected" && state.current?.document.shapes[state.selected]?.shape_type !== "rectangle") throw new Error("请先在画布选中一个矩形框");
    return {label:element("sam3Label").value,prompt:element("sam3Prompt").value.trim(),
      threshold:Number(element("sam3Threshold").value),all_labels:scope !== "label",
      ...(batch ? {regenerate:element("sam3Regenerate").checked} : {}),
      ...(scope === "selected" ? {indices:[state.selected]} : {}),
      ...(!batch ? {expected_version:state.current.version} : {})};
  }

  async function startBatch() {
    if (state.busy || generating || !await ensureSaved()) return;
    const options = {...boxOptions(true),...filters(),limit:Number(element("sam3BatchLimit").value)};
    if (!window.confirm(`按当前队列筛选，对最多 ${options.limit} 张已有检测框的图片生成轮廓候选。候选独立保存，不修改人工标注，未匹配的框原样保留。继续？`)) return;
    await api(projectUrl("/contours/jobs"), "POST", options);
    element("sam3Status").textContent = "批量轮廓任务已启动，顶部显示实时进度。完成后打开图片，点击读取本图批量候选并核对采用。";
    await pollJobs();
  }

  async function loadCandidates() {
    if (state.busy || generating || !state.current || !await ensureSaved()) return;
    const itemId = state.current.item.id;
    const projectId = state.project.id;
    const result = await api(itemUrl("/contours"));
    if (state.current?.item.id !== itemId || state.project.id !== projectId) return;
    resultItemId = itemId;
    resultProjectId = projectId;
    resultVersion = state.current.version;
    resultImageHash = state.current.image_sha256;
    resultBackend = result.backend || config?.backend || "onnx";
    resultPrompt = "检测框提示";
    resultLabel = "";
    state.sam3Candidates = result.candidates || [];
    state.sam3PreviewIndex = state.sam3Candidates.length ? 0 : -1;
    renderCandidates();
    draw();
    element("sam3Status").textContent = result.stale ? "候选对应旧版标注，已隐藏以防覆盖，请重新生成。"
      : `读取到 ${state.sam3Candidates.length} 个待核对轮廓，${result.unmatched || 0} 个框未匹配。采用后需保存 / 审核。`;
  }

  async function cancelBatch() {
    const jobs = await api(projectUrl("/jobs"));
    const job = jobs.find(job => job.kind === "sam3_contours" && ["queued", "running"].includes(job.status));
    if (!job) return toast("当前没有运行中的批量轮廓任务");
    await api(`/api/jobs/${job.id}/cancel`, "POST", {});
    element("sam3Status").textContent = "已请求取消；当前图片推理结束后停止，已完成候选保留。";
    await pollJobs();
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
  element("sam3BrowseCheckpoint").addEventListener("click", () => browse(element("sam3Backend").value === "onnx" ? "sam3_onnx" : "sam3_checkpoint", "sam3CheckpointPath").catch(error => toast(error.message, true)));
  element("sam3SaveConfig").addEventListener("click", () => saveConfig().catch(error => toast(error.message, true)));
  element("sam3Generate").addEventListener("click", () => generate().catch(error => {
    element("sam3Status").textContent = error.message;
    toast(error.message, true);
  }));
  element("sam3Release").addEventListener("click", () => release().catch(error => toast(error.message, true)));
  element("sam3Batch").addEventListener("click", () => startBatch().catch(error => toast(error.message, true)));
  element("sam3LoadCandidates").addEventListener("click", () => loadCandidates().catch(error => toast(error.message, true)));
  element("sam3CancelBatch").addEventListener("click", () => cancelBatch().catch(error => toast(error.message, true)));
  element("sam3PreviewApply").addEventListener("click", () => {
    try { applyCandidate(state.sam3PreviewIndex); } catch (error) { toast(error.message, true); }
  });
  element("sam3PreviewReturn").addEventListener("click", () => open().catch(error => toast(error.message, true)));
  element("sam3PreviewCancel").addEventListener("click", () => {
    state.sam3PreviewIndex = -1;
    renderPreview();
    draw();
  });
  element("sam3Mode").addEventListener("change", () => {
    element("sam3Generate").textContent = element("sam3Mode").value === "boxes" ? "当前图按框生成轮廓" : "生成当前图候选";
  });
  element("sam3Backend").addEventListener("change", () => {
    element("sam3CheckpointPath").value = "";
    configDirty();
  });
  element("sam3Device").addEventListener("change", configDirty);
  element("sam3Prompt").addEventListener("input", event => { event.target.dataset.edited = "true"; });
  element("sam3Label").addEventListener("change", event => {
    if (!element("sam3Prompt").dataset.edited) element("sam3Prompt").value = event.target.value.replace(/_/g, " ");
  });
  element("sam3Threshold").addEventListener("input", event => {
    element("sam3ThresholdValue").textContent = Number(event.target.value).toFixed(2);
  });
  element("sam3Dialog").addEventListener("close", () => {
    if (element("sam3Dialog").returnValue !== "preview") state.sam3PreviewIndex = -1;
    renderPreview();
    draw();
  });
  document.addEventListener("labelpilot-item-changed", renderCandidates);

  return {open, refresh, renderCandidates};
})();
