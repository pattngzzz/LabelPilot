"use strict";

const efficiency = (() => {
  const editor = {mode:"", zoomed:false, fastComplete:false, points:[], side:"positive", positive:[], negative:[],
    proposal:null, promptText:"", request:0, timer:null, start:null, interpolation:null, audit:null, showAudit:true};
  const similarity = {sequence:0, result:null, group:0, offset:0, excluded:new Set(), scanning:false};
  const audit = {job:null, active:false};
  const reasons = {missing:"疑似漏标", class_conflict:"类别冲突", box_offset:"框偏移"};
  const sides = ["上边缘", "下边缘", "左边缘", "右边缘"];

  function issueText(kinds) {
    return Object.entries(kinds || {}).map(([kind, count]) => `${reasons[kind]} ${count}`).join(" · ");
  }

  function hint(text, error=false) {
    comparisonMessage("efficiencyHint", text, error);
  }

  function frameAudit() {
    return editor.audit && editor.audit.item_id === state.current?.item.id && editor.audit.version === state.current?.version
      && editor.audit.image_sha256 === state.current?.image_sha256 ? editor.audit : null;
  }

  function render() {
    const mode = editor.mode;
    const selected = state.current?.document.shapes[state.selected];
    const count = state.current?.document.shapes.length || 0;
    const inactive = state.busy || !state.current;
    element("efficiencyMode").disabled = !state.project || state.busy;
    element("efficiencyBar").hidden = !mode;
    element("fastControls").hidden = mode !== "fast";
    element("sequenceControls").hidden = mode !== "sequence";
    element("promptControls").hidden = mode !== "prompt";
    for (const id of ["fastPrevious", "fastNext", "fastFull"]) element(id).disabled = inactive || !count;
    const keyframe = !inactive && selected?.shape_type === "rectangle";
    element("inheritNext").disabled = element("markFirstFrame").disabled = !keyframe;
    element("previewTracking").disabled = element("trackingLimit").disabled = !keyframe;
    element("previewLastFrame").disabled = !keyframe || !editor.start || editor.start.frame.id === state.current?.item.id;
    element("clearKeyframe").hidden = !editor.start;
    element("acceptPrompt").disabled = inactive || !editor.proposal;
    element("resetPrompt").disabled = inactive || !editor.positive.length && !editor.negative.length;
    element("promptPositive").setAttribute("aria-pressed", String(editor.side === "positive"));
    element("promptNegative").setAttribute("aria-pressed", String(editor.side === "negative"));

    if (mode === "fast") {
      hint(selected ? `第 ${state.selected + 1} / ${count} 框 · 数字键改类 · Del 删除`
        : editor.fastComplete ? "全图检查后按 E 审核" : "先选框，或按 ] 跳到第一框");
    } else if (mode === "sequence") {
      hint(editor.start ? `起点 ${editor.start.name} · 到最后可见帧选中同一目标`
        : "选中矩形框，可继承到同源下一张，或设置插值起点");
    } else if (mode === "fourpoint") {
      hint(`点击${sides[editor.points.length]} · Esc 取消当前四点`);
    } else if (mode === "prompt") {
      hint(editor.promptText || "点目标内部作为前景。若选多了，切到背景点后点要排除的位置。",
        Boolean(editor.promptText && !editor.proposal && !editor.promptText.endsWith("…")));
    } else hint("");

    const finding = frameAudit();
    element("auditFrameBar").hidden = !finding;
    if (finding) {
      const kinds = {};
      for (const issue of finding.findings) kinds[issue.kind] = (kinds[issue.kind] || 0) + 1;
      element("auditFrameReasons").textContent = issueText(kinds);
    }
    element("dismissAuditFrame").disabled = inactive;
    element("toggleAuditOverlay").setAttribute("aria-pressed", String(editor.showAudit));
    element("similarityButton").disabled = element("auditButton").disabled = !state.project || state.busy;
  }

  function resetPoints() {
    clearTimeout(editor.timer);
    editor.request++;
    editor.points = [];
    editor.positive = [];
    editor.negative = [];
    editor.proposal = null;
    editor.promptText = "";
  }

  function fullView() {
    editor.zoomed = false;
    selectShape(-1);
    fitImage();
  }

  function refocus() {
    const shape = state.current?.document.shapes[state.selected];
    if (editor.mode !== "fast" || !editor.zoomed || !shape || !state.image) return false;
    const [left, top, right, bottom] = bounds(shape);
    const area = viewport();
    const fit = Math.min((area.width - 16) / state.current.width, (area.height - 22) / state.current.height);
    state.scale = Math.max(fit, Math.min(12, area.width * .72 / (right - left), area.height * .70 / (bottom - top)));
    state.panX = area.width / 2 - (left + right) * state.scale / 2;
    state.panY = area.height / 2 - (top + bottom) * state.scale / 2;
    draw();
    return true;
  }

  function focusBox(index) {
    editor.fastComplete = false;
    editor.zoomed = true;
    setTool("select");
    selectShape(index);
    refocus();
    canvas.focus({preventScroll:true});
  }

  function stepBox(direction) {
    const count = state.current?.document.shapes.length || 0;
    if (!count || state.busy || state.pendingShape) return;
    const next = state.selected < 0 ? (direction > 0 ? 0 : count - 1) : Math.max(0, state.selected + direction);
    if (next >= count) {
      editor.fastComplete = true;
      fullView();
    } else focusBox(next);
  }

  function changeMode() {
    if (state.busy) return;
    editor.mode = element("efficiencyMode").value;
    editor.zoomed = editor.fastComplete = false;
    editor.side = "positive";
    resetPoints();
    setTool(editor.mode === "fourpoint" || editor.mode === "prompt" ? editor.mode : "select");
    if (editor.mode === "fast" && state.current?.document.shapes.length) focusBox(Math.max(0, state.selected));
    else fitImage();
    render();
    canvas.focus({preventScroll:true});
  }

  function toolChanged(tool) {
    if ((editor.mode === "fourpoint" || editor.mode === "prompt") && tool !== editor.mode) {
      editor.mode = "";
      element("efficiencyMode").value = "";
      resetPoints();
    }
    render();
  }

  function boxTool() {
    return editor.mode === "fourpoint" || editor.mode === "prompt" ? editor.mode : "select";
  }

  function blocksReview() {
    return editor.points.length > 0 || editor.positive.length > 0 || editor.negative.length > 0;
  }

  function keydown(event, key) {
    if (key === "escape" && blocksReview()) {
      resetPoints(); render(); draw();
      return true;
    }
    if (editor.mode !== "fast" || state.tool !== "select" || !state.current?.document.shapes.length) return false;
    if (key === "[" || key === "]" || key === "enter" && !event.target.closest("button")) {
      if (!event.repeat) stepBox(key === "[" ? -1 : 1);
      return true;
    }
    if (key === "f") { fullView(); return true; }
    if (["delete", "backspace"].includes(key) && state.selected >= 0) {
      if (event.repeat) return true;
      const index = state.selected;
      deleteShape();
      if (state.current.document.shapes.length) focusBox(Math.min(index, state.current.document.shapes.length - 1));
      else fullView();
      return true;
    }
    const replacement = /^[1-9]$/.test(key) && state.project.classes[Number(key) - 1];
    const current = state.current.document.shapes[state.selected];
    if (replacement && current) {
      if (replacement.name !== current.label) {
        checkpoint();
        current.label = replacement.name;
        state.activeClass = replacement.id;
        changed();
        renderClasses();
      }
      return true;
    }
    return false;
  }

  function handleCanvas(point) {
    if (state.tool === "fourpoint") {
      editor.points.push(point);
      if (editor.points.length < 4) { render(); draw(); return true; }
      const [top, bottom, left, right] = editor.points;
      editor.points = [];
      if (Math.abs(top[1] - bottom[1]) < 1 || Math.abs(left[0] - right[0]) < 1) {
        render(); draw(); toast("四个点未形成有效矩形，请重新点击边缘", true);
        return true;
      }
      render();
      openLabelPicker(rectanglePoints(Math.min(left[0], right[0]), Math.min(top[1], bottom[1]),
        Math.max(left[0], right[0]), Math.max(top[1], bottom[1])), point);
      return true;
    }
    if (state.tool !== "prompt") return false;
    const points = editor[editor.side];
    if (points.length >= 24) { toast("这类提示最多 24 点，可清空后重试", true); return true; }
    points.push([Math.min(point[0], state.current.width - 0.001), Math.min(point[1], state.current.height - 0.001)]);
    updatePrompt();
    return true;
  }

  function updatePrompt() {
    clearTimeout(editor.timer);
    const sequence = ++editor.request;
    const id = state.current.item.id;
    editor.proposal = null;
    editor.promptText = editor.positive.length ? "正在生成本地候选框…" : "";
    render(); draw();
    if (!editor.positive.length) return;
    editor.timer = setTimeout(async () => {
      try {
        const response = await api(itemUrl("/prompt-box"), "POST", {positive:editor.positive, negative:editor.negative,
          image_sha256:state.current.image_sha256});
        if (sequence !== editor.request || id !== state.current?.item.id || state.tool !== "prompt") return;
        editor.proposal = response.box;
        editor.promptText = "候选框已生成，可加点调整或采用";
      } catch (error) {
        if (sequence !== editor.request) return;
        editor.promptText = error.message;
      }
      render(); draw();
    }, 140);
  }

  function acceptPrompt() {
    if (!editor.proposal || state.busy) return;
    const box = editor.proposal;
    resetPoints(); render();
    openLabelPicker(rectanglePoints(...box), box.slice(2));
  }

  function drawExtras() {
    context.save();
    const finding = frameAudit();
    if (editor.showAudit && finding) {
      for (const {kind, prediction} of finding.findings) {
        pathPoints(shapePoints(prediction));
        context.strokeStyle = "#ffc27c"; context.lineWidth = 2;
        context.setLineDash([6, 4]); context.stroke();
        const [left, top] = screenPoint(bounds(prediction));
        const title = `${reasons[kind]} · ${prediction.label} ${Math.round(prediction.score * 100)}%`;
        context.font = "11px 'Segoe UI','Microsoft YaHei',sans-serif";
        context.fillStyle = "#3b2519"; context.fillRect(left, top + 2, context.measureText(title).width + 8, 16);
        context.fillStyle = "#ffc27c"; context.fillText(title, left + 4, top + 14);
      }
    }
    if (editor.proposal) {
      pathPoints(rectanglePoints(...editor.proposal));
      context.strokeStyle = "#6be7b3"; context.lineWidth = 2;
      context.setLineDash([5, 3]); context.stroke();
    }
    context.setLineDash([]);
    const points = [...editor.positive.map(point => [point, "#6be7b3", "+"]),
      ...editor.negative.map(point => [point, "#f5a4a6", "−"]),
      ...editor.points.map((point, index) => [point, "#cdbbff", String(index + 1)])];
    for (const [point, color, label] of points) {
      const [x, y] = screenPoint(point);
      context.beginPath(); context.arc(x, y, 7.5, 0, Math.PI * 2);
      context.fillStyle = color; context.fill();
      context.font = "bold 10px sans-serif"; context.fillStyle = "#10131a";
      context.textAlign = "center"; context.fillText(label, x, y + 3.5);
    }
    context.restore();
  }

  async function savedAnchor() {
    const selected = state.current?.document.shapes[state.selected];
    if (state.busy || selected?.shape_type !== "rectangle") return null;
    if (state.dirty && !await saveCurrent("draft")) return null;
    return {id:state.current.item.id, version:state.current.version, shape:state.selected,
      image_sha256:state.current.image_sha256, source_fingerprint:state.current.source_fingerprint};
  }

  async function inherit() {
    const frame = await savedAnchor();
    if (!frame) return;
    setBusy(true, "正在继承到同源下一张…");
    let result;
    try {
      result = await api(projectUrl("/sequence/inherit"), "POST", frame);
      await refreshList();
    } finally { setBusy(false); }
    await loadItem(result.id);
    toast(result.appended ? "已在下一张追加草稿框，请检查位置" : "下一张已有匹配框，已打开供检查");
  }

  function markStart(frame) {
    if (!frame) return;
    editor.start = {frame, name:state.current.item.photo_id};
    render();
  }

  function showInterpolatedFrame() {
    const plan = editor.interpolation?.result;
    const target = plan?.targets.find(item => item.id === element("sequenceTarget").value);
    const preview = element("sequencePreview");
    preview.hidden = !target;
    preview.replaceChildren();
    if (!target) return;
    const scene = document.createElement("div");
    scene.className = "comparison-frame-scene";
    scene.style.setProperty("--image-ratio", target.width / target.height);
    const img = document.createElement("img");
    img.src = projectUrl(`/items/${target.id}/image?thumbnail=true`);
    img.alt = target.photo_id;
    const overlay = document.createElement("div");
    overlay.className = "comparison-frame-overlay";
    overlay.append(comparisonBox(target.box, target, "matched", plan.label));
    scene.append(img, overlay);
    preview.append(scene);
  }

  function showSequencePlan(kind, request, result, summary) {
    editor.interpolation = {kind, request, result};
    const count = result.targets.length;
    element("sequenceTitle").textContent = kind === "tracking" ? "跟踪辅助" : "关键帧插值";
    element("sequenceSummary").textContent = summary;
    element("sequenceTargetLabel").hidden = !count;
    element("sequenceTarget").replaceChildren(...result.targets.map(row => new Option(row.photo_id, row.id)));
    element("sequenceStop").hidden = !result.stop;
    element("sequenceStopText").textContent = result.stop ? `${result.stop.photo_id} · ${result.stop.reason}` : "";
    element("sequenceApply").disabled = !count;
    comparisonMessage("sequenceStatus", "");
    showInterpolatedFrame();
    element("sequenceDialog").showModal();
  }

  async function previewInterpolation() {
    const end = await savedAnchor();
    if (!end || !editor.start) return;
    const request = {start:editor.start.frame, end};
    setBusy(true, "正在核对区间中的标注…");
    try {
      const result = await api(projectUrl("/sequence/preview"), "POST", request);
      showSequencePlan("sequence", request, result, `${editor.start.name} → ${state.current.item.photo_id} · 预计写入 ${result.targets.length} 张草稿。已有匹配框 ${result.skipped.matched} 张，已审核或跳过 ${result.skipped.status} 张，不可用 ${result.skipped.unreadable} 张。`);
    } finally { setBusy(false); }
  }

  async function previewTracking() {
    const start = await savedAnchor();
    if (!start) return;
    const request = {start, limit:Number(element("trackingLimit").value)};
    setBusy(true, "正在逐帧跟踪所选目标…");
    try {
      const result = await api(projectUrl("/tracking/preview"), "POST", request);
      showSequencePlan("tracking", request, result,
        `从 ${state.current.item.photo_id} 跟踪 · 可写入 ${result.targets.length} 张草稿 · 已有 ${result.skipped} 张，保留原标注。`);
    } finally { setBusy(false); }
  }

  async function applyInterpolation() {
    const plan = editor.interpolation;
    if (!plan?.result.targets.length || element("sequenceApply").disabled) return;
    element("sequenceApply").disabled = true;
    try {
      const result = await api(projectUrl(`/${plan.kind}/apply`), "POST", {...plan.request, token:plan.result.token});
      editor.start = null;
      if (result.stop) {
        plan.applied = true;
        comparisonMessage("sequenceStatus", `已保存 ${result.targets.length} 张草稿，可继续打开待修帧。`);
      } else {
        editor.interpolation = null;
        element("sequenceDialog").close();
      }
      await refreshList();
      render();
      toast(`已生成 ${result.targets.length} 张中间帧草稿，可进入待审核队列检查`);
    } catch (error) {
      comparisonMessage("sequenceStatus", error.message, true);
    } finally { element("sequenceApply").disabled = Boolean(plan.applied); }
  }

  function resetProject() {
    editor.start = editor.interpolation = editor.audit = null;
    editor.zoomed = editor.fastComplete = false;
    editor.mode = "";
    element("efficiencyMode").value = "";
    resetPoints();
    setTool("select");
    audit.job = null;
  }

  function itemChanged() {
    resetPoints();
    editor.audit = null;
    editor.showAudit = true;
    editor.zoomed = editor.fastComplete = false;
    render();
    if (editor.mode === "fast" && state.current?.document.shapes.length) focusBox(0);
    else draw();
    if (state.current && filters().status === "audit") {
      const current = state.current;
      api(itemUrl("/audit")).then(finding => {
        if (state.current !== current || filters().status !== "audit") return;
        editor.audit = finding ? {...finding, item_id:current.item.id} : null;
        render(); draw();
      }).catch(error => toast(error.message, true));
    }
  }

  async function enterQueue(status, batch) {
    element("batchSelect").value = batch;
    element("statusSelect").value = status;
    element("searchInput").value = "";
    localStorage.setItem(`labelpilot-batch-${state.project.id}`, batch);
    state.offset = 0;
    await refreshList();
    if (state.items[0]) await loadItem(state.items[0].id);
    else finishReviewQueue();
  }

  function showSimilarity() {
    if (!state.project) return;
    element("similarityBatch").innerHTML = element("exportBatch").innerHTML;
    element("similarityBatch").value = filters().batch;
    clearSimilarity();
    element("similarityDialog").showModal();
  }

  function clearSimilarity() {
    similarity.sequence++;
    similarity.result = null;
    similarity.scanning = false;
    element("scanSimilarity").disabled = false;
    element("similarityResults").hidden = true;
    element("applySimilarity").disabled = true;
    element("similaritySelectionInfo").textContent = "";
    comparisonMessage("similarityStatus", "");
  }

  function updateSimilaritySelection() {
    const {result, excluded} = similarity;
    element("similaritySelectionInfo").textContent = `队列保留 ${result.images - excluded.size} 张 · 暂不进入队列 ${excluded.size} 张`;
  }

  function renderSimilarityGroup() {
    const {result, group, offset, excluded} = similarity;
    updateSimilaritySelection();
    const selected = result.groups[group];
    element("similarityResults").hidden = !selected;
    if (!selected) return;
    element("similarityGroupInfo").textContent = `第 ${group + 1} / ${result.groups.length} 组 · ${selected.members.length} 张`;
    element("similarityPrevGroup").disabled = !group;
    element("similarityNextGroup").disabled = group + 1 >= result.groups.length;
    element("similarityPages").hidden = selected.members.length <= 16;
    element("similarityPrevPage").disabled = !offset;
    element("similarityNextPage").disabled = offset + 16 >= selected.members.length;
    element("similarityPageInfo").textContent = `${offset + 1}–${Math.min(offset + 16, selected.members.length)} / ${selected.members.length}`;
    element("similarityCards").innerHTML = selected.members.slice(offset, offset + 16).map(image =>
      `<label class="similarity-card" title="${escapeHtml(image.relative)}">
        <img loading="lazy" src="${projectUrl(`/items/${image.id}/image?thumbnail=true`)}" alt="">
        <span><input type="checkbox" data-similarity-id="${image.id}" ${excluded.has(image.id) ? "" : "checked"}>
          <span>${escapeHtml(image.photo_id)}<small>${image.width} × ${image.height}${image.id === selected.representative ? " · 推荐" : ""}</small></span></span>
      </label>`).join("");
  }

  async function scanSimilarity() {
    if (similarity.scanning) return;
    const sequence = ++similarity.sequence;
    similarity.scanning = true;
    element("scanSimilarity").disabled = element("applySimilarity").disabled = true;
    element("similarityResults").hidden = true;
    element("similaritySelectionInfo").textContent = "";
    const batch = element("similarityBatch").value;
    comparisonMessage("similarityStatus", "正在读取本机缩略图并计算相似度…");
    const active = () => sequence === similarity.sequence && element("similarityDialog").open;
    try {
      const result = await api(projectUrl("/similarity/scan"), "POST", {batch, level:element("similarityLevel").value});
      if (!active()) return;
      const memberIds = new Set(result.groups.flatMap(group => group.members.map(item => item.id)));
      Object.assign(similarity, {result, group:0, offset:0, excluded:new Set(result.excluded.filter(id => memberIds.has(id)))});
      comparisonMessage("similarityStatus", `扫描 ${result.images} 张未标注图 · 找到 ${result.groups.length} 组相似候选${result.unreadable ? ` · ${result.unreadable} 张读图失败` : ""}`);
      renderSimilarityGroup();
      element("applySimilarity").disabled = !result.images;
    } catch (error) {
      if (active()) comparisonMessage("similarityStatus", error.message, true);
    } finally {
      if (active()) { similarity.scanning = false; element("scanSimilarity").disabled = false; }
    }
  }

  function advanceSimilarity(delta, groups=false) {
    if (groups) { similarity.group += delta; similarity.offset = 0; }
    else similarity.offset += delta * 16;
    renderSimilarityGroup();
  }

  async function applySimilarity() {
    const result = similarity.result;
    if (!result || element("applySimilarity").disabled || !await ensureSaved()) return;
    element("applySimilarity").disabled = true;
    try {
      await api(projectUrl("/similarity/selection"), "POST", {batch:result.batch, excluded:[...similarity.excluded]});
      element("similarityDialog").close();
      await enterQueue("curated", result.batch);
    } catch (error) {
      comparisonMessage("similarityStatus", error.message, true);
    } finally { element("applySimilarity").disabled = false; }
  }

  function updateJobs(jobs) {
    audit.job = jobs.find(job => job.kind === "audit");
    audit.active = jobs.some(job => ["queued", "running"].includes(job.status));
    if (!element("auditDialog").open) return;
    const job = audit.job;
    const active = job && ["queued", "running"].includes(job.status);
    const progress = element("auditProgress");
    progress.hidden = !active;
    element("stopAudit").hidden = !active;
    element("startAudit").disabled = audit.active;
    const path = state.project.model_path;
    element("auditModelName").textContent = path ? path.split(/[\\/]/).at(-1) : "请先保存本地检测模型";
    if (!job) return comparisonMessage("auditStatus", "");
    const checked = job.done + job.failed + job.skipped;
    if (active) { progress.max = job.total; progress.value = checked; }
    const status = {queued:"等待中", running:"正在扫描", completed:"扫描完成", completed_with_errors:"扫描完成，部分失败",
      failed:"扫描失败", cancelled:"已取消", interrupted:"上次扫描中断"}[job.status];
    comparisonMessage("auditStatus", `${status} · ${checked} / ${job.total} 张 · 疑点 ${job.flagged} 张${job.failed ? ` · 失败 ${job.failed} 张` : ""}${job.errors?.length ? `。 ${job.errors.at(-1).error}` : ""}`, job.status === "failed");
  }

  async function showAudit() {
    if (!state.project) return;
    element("auditBatch").innerHTML = element("exportBatch").innerHTML;
    element("auditBatch").value = filters().batch;
    element("auditDialog").showModal();
    await pollJobs();
  }

  async function runAudit() {
    if (element("startAudit").disabled || !await ensureSaved()) return;
    for (const field of [element("auditConfidence"), element("auditLimit")]) if (!field.reportValidity()) return;
    element("startAudit").disabled = true;
    try {
      await api(projectUrl("/audits"), "POST", {batch:element("auditBatch").value,
        confidence:element("auditConfidence").valueAsNumber, limit:element("auditLimit").valueAsNumber, image_size:640});
      await pollJobs();
    } catch (error) {
      comparisonMessage("auditStatus", error.message, true);
      element("startAudit").disabled = false;
    }
  }

  async function dismissAudit() {
    const finding = frameAudit();
    if (!finding || !await ensureSaved()) return;
    const index = state.items.findIndex(item => item.id === state.current.item.id);
    await api(itemUrl("/audit/dismiss"), "POST", {id:finding.id});
    editor.audit = null;
    render(); draw();
    await refreshList();
    const next = state.items[index] || state.items.at(-1);
    if (next) await loadItem(next.id);
    else finishReviewQueue();
  }

  function install() {
    listen("efficiencyMode", "change", changeMode);
    listen("fastPrevious", "click", () => stepBox(-1));
    listen("fastNext", "click", () => stepBox(1));
    listen("fastFull", "click", fullView);
    listen("inheritNext", "click", inherit);
    listen("markFirstFrame", "click", async () => markStart(await savedAnchor()));
    listen("previewLastFrame", "click", previewInterpolation);
    listen("previewTracking", "click", previewTracking);
    listen("openTrackStop", "click", async () => {
      const stop = editor.interpolation?.result.stop;
      if (!stop) return;
      element("sequenceDialog").close();
      await loadItem(stop.id);
    });
    listen("clearKeyframe", "click", () => { editor.start = null; render(); });
    listen("sequenceTarget", "change", showInterpolatedFrame);
    listen("sequenceApply", "click", applyInterpolation);
    for (const [id, side] of [["promptPositive","positive"], ["promptNegative","negative"]]) {
      listen(id, "click", () => { editor.side = side; render(); canvas.focus({preventScroll:true}); });
    }
    listen("resetPrompt", "click", () => { resetPoints(); render(); draw(); });
    listen("acceptPrompt", "click", acceptPrompt);
    listen("similarityButton", "click", showSimilarity);
    listen("scanSimilarity", "click", scanSimilarity);
    listen("similarityBatch", "change", clearSimilarity);
    listen("similarityLevel", "change", clearSimilarity);
    listen("similarityDialog", "close", () => { if (!element("similarityDialog").open) clearSimilarity(); });
    listen("similarityPrevGroup", "click", () => advanceSimilarity(-1, true));
    listen("similarityNextGroup", "click", () => advanceSimilarity(1, true));
    listen("similarityPrevPage", "click", () => advanceSimilarity(-1));
    listen("similarityNextPage", "click", () => advanceSimilarity(1));
    listen("similarityKeepAll", "click", () => { similarity.excluded.clear(); renderSimilarityGroup(); });
    listen("similarityKeepRecommended", "click", () => {
      similarity.excluded = new Set(similarity.result.groups.flatMap(group => group.members
        .filter(member => member.id !== group.representative).map(member => member.id)));
      renderSimilarityGroup();
    });
    listen("similarityCards", "change", event => {
      const id = event.target.dataset.similarityId;
      if (!id) return;
      if (event.target.checked) similarity.excluded.delete(id);
      else similarity.excluded.add(id);
      updateSimilaritySelection();
    });
    listen("applySimilarity", "click", applySimilarity);
    listen("auditButton", "click", showAudit);
    listen("auditModelSettings", "click", openModelDialog);
    listen("modelDialog", "close", () => { if (element("auditDialog").open) return pollJobs(); });
    listen("startAudit", "click", runAudit);
    listen("stopAudit", "click", async () => {
      if (audit.job) await api(`/api/jobs/${audit.job.id}/cancel`, "POST", {});
      await pollJobs();
    });
    listen("openAuditQueue", "click", async () => {
      if (!await ensureSaved()) return;
      const batch = element("auditBatch").value;
      element("auditDialog").close();
      await enterQueue("audit", batch);
    });
    listen("toggleAuditOverlay", "click", () => { editor.showAudit = !editor.showAudit; render(); draw(); });
    listen("dismissAuditFrame", "click", dismissAudit);
  }

  return {install, render, boxTool, toolChanged, keydown, handleCanvas, drawExtras, refocus, resetProject,
    itemChanged, updateJobs, issueText, blocksReview, enterQueue};
})();
