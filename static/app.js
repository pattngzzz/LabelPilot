"use strict";

const element = id => document.getElementById(id);
const copy = value => structuredClone(value);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
const statusNames = {unstarted:"未开始",imported:"人工标注 · 待审核",predicted:"模型待审核",draft:"编辑中",reviewed:"已审核",skipped:"已跳过",error:"读取异常"};
const reviewModes = new Set(["review", "review_empty", "review_difficult", "review_low"]);
const state = {projects:[],project:null,items:[],filtered:0,offset:0,pageSize:60,current:null,image:null,selected:-1,
  activeClass:0,tool:"select",dirty:false,busy:false,undo:[],redo:[],scale:1,panX:0,panY:0,drag:null,
  polygon:[],rectangle:null,pendingShape:null,labelPickerSaving:false,canvasPointer:null,space:false,showLabels:true,prediction:null,loadSequence:0,listSequence:0};
const canvas = element("canvas");
const context = canvas.getContext("2d");
const sessionToken = document.querySelector('meta[name="labelpilot-token"]').content;
const modelBrowser = {mode:"file",target:"modelPath",listing:null,loading:false,sequence:0};
const exportPreview = {sequence:0, timer:null, plan:null, running:false};
const exportSplitLabels = {train:"训练", val:"验证", test:"测试"};
const exportSplitHints = {
  source_group:"任务清单里同一来源组的图片会进入同一个集合。实际比例取决于各组的图片数量。",
  directory:"同一原图文件夹的图片会进入同一个集合。若同源图分散在不同文件夹，请使用任务清单分组。",
  image:"逐张随机分配。同一视频的相邻帧或重复图片可能跨集。"
};

const imageBrowser = {mode:"folder",target:"newImageRoot",listing:null,selected:new Set(),loading:false,valid:false,sequence:0};
const projectImages = {files:[]};
const exportAugmentation = {presets:[],maxCopies:3,loaded:false};
let searchTimer;
let jobTimer;
let trainingTimer;
let trainingSequence = 0;
let trainingSubmitting = false;
let trainingRuns = [];
const comparisonUi = {project:null, timer:null, session:0, read:0, sets:[], runs:[], selected:"",
  report:null, reportData:null, loadingReport:null, submitting:false, freezing:false};
const comparisonReview = {url:"", listSequence:0, imageSequence:0, page:null, frame:null, imageUrl:null};

function toast(message, error=false) {
  const node = document.createElement("div");
  node.className = "toast" + (error ? " error" : "");
  node.textContent = message;
  element("toasts").append(node);
  setTimeout(() => node.remove(), error ? 9000 : 4500);
}

function listen(id, event, handler) {
  element(id).addEventListener(event, (...args) => {
    Promise.resolve().then(() => handler(...args)).catch(error => toast(error.message, true));
  });
}

async function api(path, method="GET", body) {
  const response = await fetch(path, {method, headers:{"Content-Type":"application/json","X-LabelPilot-Token":sessionToken},
    body:body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json().catch(() => ({detail:"服务返回异常，请查看启动窗口"}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

function projectUrl(suffix="") { return `/api/projects/${state.project.id}${suffix}`; }
function itemUrl(suffix="") { return projectUrl(`/items/${state.current.item.id}${suffix}`); }
function filters() { return {query:element("searchInput").value.trim(),batch:element("batchSelect").value,status:element("statusSelect").value}; }
function managedQueue(status=filters().status) { return reviewModes.has(status) || ["audit","curated","triage","spot_review"].includes(status); }
function setBusy(value, message="") {
  state.busy = value;
  for (const id of ["predictButton","singlePredict","batchPredict","saveModel","saveButton","reviewButton","skipButton","previousImage","nextImage","projectSelect","deleteShape","shapeClass","shapeDescription","shapeDifficult","newLabelInput","addLabelButton"]) element(id).disabled = value;
  element("connectionStatus").textContent = value ? message || "正在处理…" : "本地工作区 · 仅本机可访问";
  efficiency.render();
  updateCrosshair();
}

function setDirty(value=true) {
  state.dirty = value;
  element("dirtyBadge").hidden = !value;
  if (value && state.current) { setStatus("draft"); workspaceResume.edited(); }
}

function setStatus(status) {
  element("statusBadge").className = `pill ${status}`;
  element("statusBadge").textContent = statusNames[status] || status;
}

function showCanvasMessage(title, text="") {
  element("canvasMessage").hidden = !title;
  element("canvasMessage").querySelector("h3").textContent = title;
  element("canvasMessage").querySelector("p").textContent = text;
}

async function initialize() {
  const response = await api("/api/projects");
  state.projects = response.projects;
  for (const failure of response.errors) toast(`${failure.name}：${failure.error}`, true);
  element("projectSelect").innerHTML = state.projects.map(project => `<option value="${escapeHtml(project.id)}">${escapeHtml(project.name)}</option>`).join("");
  if (!state.projects.length) {
    showCanvasMessage("接入你的第一个数据集", "点击左上角“选择图片”，选择本机图片文件或文件夹");
    element("projectDialog").showModal();
    return;
  }
  const previous = localStorage.getItem("labelpilot-project");
  await selectProject(state.projects.find(project => project.id === previous)?.id || state.projects[0].id);
  api("/api/system").then(system => {
    element("modelRuntime").textContent = system.torch ? `运行设备 ${system.device} · PyTorch ${system.torch}` : "当前 Python 缺少 PyTorch";
    element("modelCaption").textContent = system.torch ? `${system.cuda ? "CUDA" : "CPU"} · 本地模型 · 按需加载` : "需要本地模型环境";
  }).catch(error => toast(error.message, true));
}

async function selectProject(projectId) {
  if (state.busy) return;
  if (!await ensureSaved()) { element("projectSelect").value = state.project.id; return; }
  state.project = state.projects.find(project => project.id === projectId);
  state.current = null;
  state.image = null;
  state.selected = -1;
  state.undo = [];
  state.redo = [];
  state.offset = 0;
  efficiency.resetProject();
  state.activeClass = Math.max(0, state.project.classes.findIndex(item => item.name === "sample_container"));
  localStorage.setItem("labelpilot-project", projectId);
  element("projectSelect").value = projectId;
  element("projectSubtitle").textContent = `${state.project.batches.length} 个批次 · ${state.project.classes.length} 个标签 · 原始数据只读`;
  const options = '<option value="">全部批次</option>' + state.project.batches.map(batch => `<option value="${escapeHtml(batch)}">${escapeHtml(batch)}</option>`).join("");
  element("batchSelect").innerHTML = options;
  element("exportBatch").innerHTML = options;
  element("exportSplitUnit").value = state.project.has_source_groups ? "source_group" : "image";
  const rememberedBatch = localStorage.getItem(`labelpilot-batch-${projectId}`);
  element("batchSelect").value = state.project.batches.includes(rememberedBatch) ? rememberedBatch : state.project.batches.includes("B001") ? "B001" : "";
  element("searchInput").value = "";
  element("statusSelect").value = "";
  element("modelPath").value = state.project.model_path;
  const remembered = workspaceResume.position();
  workspaceResume.restore(remembered);
  renderClasses();
  await refreshList();
  const prior = state.items.find(item => item.id === remembered?.item);
  if (state.items.length) await loadItem((prior || state.items[0]).id, prior ? remembered : null);
  else showCanvasMessage("当前筛选没有图片", "切换批次或审核状态");
  clearInterval(jobTimer);
  jobTimer = setInterval(() => pollJobs().catch(() => {}), 3000);
  await pollJobs();
}

async function refreshList() {
  if (!state.project) return;
  const sequence = ++state.listSequence;
  const values = filters();
  const response = await api(projectUrl("/items?") + new URLSearchParams({...values,offset:state.offset,limit:state.pageSize}));
  if (sequence !== state.listSequence) return;
  if (state.offset && state.offset >= response.filtered) {
    state.offset = Math.max(0, Math.floor((response.filtered - 1) / state.pageSize) * state.pageSize);
    return refreshList();
  }
  state.items = response.items;
  state.filtered = response.filtered;
  element("filteredCount").textContent = response.filtered.toLocaleString();
  element("totalCount").textContent = response.total.toLocaleString();
  element("predictedCount").textContent = response.counts.predicted.toLocaleString();
  element("reviewedCount").textContent = response.counts.reviewed.toLocaleString();
  const percent = response.total ? response.counts.reviewed / response.total * 100 : 0;
  element("progressText").textContent = `${percent.toFixed(1)}%`;
  element("reviewProgress").style.width = `${percent}%`;
  element("progressCaption").textContent = `${response.counts.imported} 张人工标注待审核 · ${response.counts.unstarted} 张未开始`;
  element("pageInfo").textContent = response.filtered ? `${state.offset + 1}–${Math.min(state.offset + state.pageSize,response.filtered)} / ${response.filtered}` : "0 / 0";
  element("prevPage").disabled = state.offset === 0;
  element("nextPage").disabled = state.offset + state.pageSize >= response.filtered;
  renderQueue();
  workspaceResume.prefetch();
}

function renderQueue() {
  const review = reviewModes.has(filters().status);
  element("reviewQueueButton").setAttribute("aria-pressed", String(review));
  element("imageList").innerHTML = state.items.map(item => {
    const reasons = [];
    if (item.review?.empty) reasons.push("待确认空图");
    if (item.review?.difficult) reasons.push("难例");
    if (item.review?.low_confidence) reasons.push(`低置信 ${Math.round(item.review.confidence * 100)}%`);
    if (item.audit) reasons.push(efficiency.issueText(item.audit.kinds));
    if (item.route) reasons.push(automationUi.routeText(item.route));
    return `<button class="image-row ${state.current?.item.id === item.id ? "selected" : ""}" data-item="${item.id}" title="${escapeHtml(item.relative)}">
      <img loading="lazy" src="${projectUrl(`/items/${item.id}/image?thumbnail=true`)}" alt="">
      <span class="image-row-info"><strong>${escapeHtml(item.photo_id)}</strong><small>${statusNames[item.status]} · ${item.count} 对象</small>
      ${reasons.length ? `<small class="review-reasons">${reasons.join(" · ")}</small>` : ""}</span><i class="row-dot ${item.status}"></i>
    </button>`;
  }).join("") || `<p class="empty-list">${review ? "当前范围没有待审核图片" : "当前筛选暂无图片"}</p>`;
}

function finishReviewQueue() {
  const status = filters().status;
  const titles = {audit:"标注疑点", curated:"筛后队列", triage:"智能审核", spot_review:"按来源抽样复核"};
  const messages = {audit:"当前范围没有待复核疑点", curated:"当前范围没有筛后的未标注图",
    triage:"当前范围没有已分流的待审核图", spot_review:"当前范围没有待复核样本"};
  const title = titles[status] || "待审核队列";
  const message = messages[status] || "当前范围没有待审核图片";
  Object.assign(state, {current:null, image:null, selected:-1, undo:[], redo:[], drag:null,
    rectangle:null, polygon:[], pendingShape:null});
  element("imageTitle").textContent = title;
  element("imageMeta").textContent = "";
  element("sourceNote").textContent = "";
  setStatus("");
  showCanvasMessage(message, "切换批次或审核筛选继续");
  renderObjects();
  efficiency.itemChanged();
}

async function ensureSaved() {
  await workspaceResume.wait();
  if (!state.dirty) return true;
  if (!window.confirm("当前图片有未保存修改。确定保存草稿后继续？取消将留在当前图片。")) return false;
  return saveCurrent("draft");
}

async function loadItem(itemId, remembered=null) {
  if (state.busy || !await ensureSaved()) return;
  const sequence = ++state.loadSequence;
  setBusy(true,"读取原图和标注…");
  showCanvasMessage("正在载入原图…");
  try {
    const current = await api(projectUrl(`/items/${itemId}`));
    const image = await workspaceResume.imageFor(current);
    if (sequence !== state.loadSequence) return;
    state.current = current;
    state.image = image;
    state.selected = -1;
    state.undo = [];
    state.redo = [];
    state.drag = null;
    state.rectangle = null;
    state.polygon = [];
    setDirty(false);
    setStatus(current.status);
    element("imageTitle").textContent = current.item.photo_id;
    element("imageMeta").textContent = `${current.width} × ${current.height} · ${current.item.batch}`;
    element("sourceNote").textContent = current.item.note || (current.item.source_kind === "human" ? "已载入原生人工标注；属性保留，尚未独立审核" : current.item.source_kind === "model" ? "已载入历史模型预标注；必须人工审核" : "原图只读 · 新标注独立保存");
    element("sourceNote").title = current.item.relative;
    showCanvasMessage("");
    renderQueue();
    renderObjects();
    fitImage();
    efficiency.itemChanged();
    workspaceResume.loaded(remembered);
  } catch (error) {
    showCanvasMessage("图片载入失败",error.message);
    throw error;
  } finally { setBusy(false); }
}

async function saveCurrent(status="draft") {
  await workspaceResume.wait();
  if (!state.current || state.busy) return false;
  let confirmedEmpty = false;
  if (status === "reviewed" && state.current.document.shapes.length === 0) {
    confirmedEmpty = window.confirm("此图没有任何标注。你是否已逐图检查，确认确实不存在项目类别目标？确定后将作为人工确认的负样本。模型零检测不等于负样本。");
    if (!confirmedEmpty) return false;
  }
  setBusy(true,"正在安全保存…");
  try {
    const response = await api(itemUrl(),"PUT",{document:state.current.document,expected_version:state.current.version,
      status,confirmed_empty:confirmedEmpty,image_sha256:state.current.image_sha256,source_fingerprint:state.current.source_fingerprint});
    Object.assign(state.current,response);
    setDirty(false);
    workspaceResume.savedManually();
    setStatus(response.status);
    draw();
    toast(status === "reviewed" ? "已人工确认并保存" : status === "skipped" ? "已跳过；标注仍保留" : "草稿已保存，原始标注未改动");
    await refreshList();
    return true;
  } catch (error) {
    toast(error.message,true);
    return false;
  } finally { setBusy(false); }
}

async function navigateImage(direction, reviewed=false) {
  if (!state.current || state.busy) return;
  if (reviewed && (state.rectangle || state.polygon.length || state.pendingShape || efficiency.blocksReview())) return toast("请先完成或取消当前绘制", true);
  const previousIds = state.items.map(item => item.id);
  const previousOffset = state.offset;
  const currentIndex = previousIds.indexOf(state.current.item.id);
  const nextId = previousIds[currentIndex + direction];
  if (reviewed && !await saveCurrent("reviewed")) return;
  if (!reviewed && state.dirty && !await saveCurrent("draft")) return;
  if (reviewed && managedQueue()) {
    const stillNext = state.items.find(item => item.id === nextId);
    const next = stillNext || (state.offset === previousOffset && state.items[currentIndex]) || state.items[0];
    if (next) return loadItem(next.id);
    return finishReviewQueue();
  }
  if (nextId) return loadItem(nextId);
  if (direction > 0 && state.offset + state.pageSize < state.filtered) {
    state.offset += state.pageSize;
    await refreshList();
    if (state.items[0]) return loadItem(state.items[0].id);
  }
  if (direction < 0 && state.offset > 0) {
    state.offset = Math.max(0,state.offset - state.pageSize);
    await refreshList();
    if (state.items.length) return loadItem(state.items.at(-1).id);
  }
  if (reviewed && state.items[0] && !state.items.some(item => item.id === state.current.item.id)) return loadItem(state.items[0].id);
  toast("已到当前队列边界，可切换批次继续");
}

function snapshot() { return copy(state.current.document); }
function checkpoint(before) {
  if (!state.current) return;
  state.undo.push(before || snapshot());
  if (state.undo.length > 100) state.undo.shift();
  state.redo = [];
}
function changed() { setDirty(); renderObjects(); draw(); }
function undo(redo=false) {
  if (!state.current || state.busy) return;
  const source = redo ? state.redo : state.undo;
  if (!source.length) return;
  (redo ? state.undo : state.redo).push(snapshot());
  state.current.document = source.pop();
  state.selected = -1;
  changed();
}

function classInfo(label) { return state.project.classes.find(item => item.name === label) || {name:label,display:label,color:"#f87171"}; }
function renderClasses() {
  element("classList").innerHTML = state.project.classes.map((item,index) => `<button class="class-row ${state.activeClass === index ? "selected" : ""}" data-class="${index}" title="${escapeHtml(item.name)}"><span class="class-swatch" style="background:${item.color}"></span><span>${escapeHtml(item.name)}<small>ID ${item.id}</small></span>${index < 9 ? `<kbd>${index + 1}</kbd>` : ""}</button>`).join("");
  const previousLabel = element("shapeClass").value;
  element("shapeClass").innerHTML = state.project.classes.map(item => `<option value="${escapeHtml(item.name)}">${item.id} · ${escapeHtml(item.name)}</option>`).join("");
  const selectedLabel = state.current?.document.shapes[state.selected]?.label;
  element("shapeClass").value = selectedLabel || previousLabel || state.project.classes[state.activeClass].name;
  element("activeClassText").textContent = `绘制标签：${state.project.classes[state.activeClass].name}`;
}

async function addCustomLabel() {
  if (!state.project || state.busy) return;
  const name = element("newLabelInput").value.trim();
  if (!name) return toast("请输入标签名称", true);
  const projectId = state.project.id;
  element("addLabelButton").disabled = true;
  element("newLabelInput").disabled = true;
  try {
    const project = await api(projectUrl("/labels"), "POST", {name});
    if (state.project.id !== projectId) return;
    Object.assign(state.project, project);
    state.activeClass = project.classes.findIndex(item => item.name === name);
    const selected = state.current?.document.shapes[state.selected];
    if (selected && selected.label !== name) {
      checkpoint();
      selected.label = name;
      changed();
    }
    renderClasses();
    renderObjects();
    element("newLabelInput").value = "";
    element("newLabelInput").blur();
    element("projectSubtitle").textContent = `${state.project.batches.length} 个批次 · ${state.project.classes.length} 个标签 · 原始数据只读`;
    toast(selected ? `标签 ${name} 已设为当前对象标签；修改尚未保存` : `标签 ${name} 已就绪，按 R 画框`);
  } finally {
    element("addLabelButton").disabled = false;
    element("newLabelInput").disabled = false;
  }
}

function renderObjects() {
  const shapes = state.current?.document.shapes || [];
  element("objectCount").textContent = shapes.length;
  element("annotationSummary").textContent = shapes.length ? `${shapes.length} 个对象 · 点击选择 / 调整` : "暂无对象 · R 画框或运行模型";
  element("objectList").innerHTML = shapes.map((shape,index) => {
    const info = classInfo(shape.label);
    return `<button class="object-row ${state.selected === index ? "selected" : ""}" data-shape="${index}" title="${escapeHtml(shape.label)}"><span class="object-index">${index + 1}</span><i class="class-swatch" style="background:${info.color}"></i><span class="object-name">${escapeHtml(info.name)}</span><small>${shape.score == null ? "人工" : `${Math.round(shape.score*100)}%`}</small></button>`;
  }).join("") || '<p class="empty-list">模型先标，你来把关<br>或按 R 开始画框</p>';
  const selected = shapes[state.selected];
  element("shapeProperties").hidden = !selected;
  if (selected) {
    element("shapeClass").value = selected.label;
    element("shapeDescription").value = selected.description || "";
    element("shapeDifficult").checked = Boolean(selected.difficult);
  }
  element("undoButton").disabled = !state.undo.length;
  element("redoButton").disabled = !state.redo.length;
  efficiency.render();
}

function selectShape(index) {
  state.selected = index;
  const shape = state.current?.document.shapes[index];
  if (shape) {
    const classIndex = state.project.classes.findIndex(item => item.name === shape.label);
    if (classIndex >= 0) state.activeClass = classIndex;
  }
  renderClasses(); renderObjects(); draw();
}

function positionLabelPicker() {
  if (!state.pendingShape || !element("labelPicker").open) return;
  const dialog = element("labelPicker");
  const canvasBounds = canvas.getBoundingClientRect();
  const anchor = screenPoint(state.pendingShape.anchor);
  const size = dialog.getBoundingClientRect();
  const horizontal = canvasBounds.left + anchor[0];
  const vertical = canvasBounds.top + anchor[1];
  const preferredLeft = horizontal + 14 + size.width <= innerWidth - 12 ? horizontal + 14 : horizontal - size.width - 14;
  dialog.style.left = `${Math.max(12, Math.min(innerWidth - size.width - 12, preferredLeft))}px`;
  dialog.style.top = `${Math.max(12, Math.min(innerHeight - size.height - 12, vertical))}px`;
}

function renderLabelOptions() {
  const query = element("pickerLabelInput").value.trim();
  const labels = state.project.classes.filter(item => item.name.toLowerCase().includes(query.toLowerCase()));
  element("pickerLabelOptions").innerHTML = labels.map(item => `<button type="button" class="picker-label-option ${item.id === state.activeClass ? "selected" : ""}" data-label="${escapeHtml(item.name)}"><i class="class-swatch" style="background:${item.color}"></i><span>${escapeHtml(item.name)}</span><small>ID ${item.id}</small></button>`).join("") || '<p class="picker-empty">没有匹配的已有标签，可直接创建新标签。</p>';
  const existing = state.project.classes.some(item => item.name === query);
  element("confirmPickedLabel").textContent = !query ? "使用当前标签" : existing ? "使用此标签" : "创建并使用";
  element("pickerCurrentLabel").textContent = `当前标签：${state.project.classes[state.activeClass].name}`;
  positionLabelPicker();
}

function openLabelPicker(points, anchor) {
  state.pendingShape = {points, anchor};
  element("pickerLabelInput").value = "";
  element("pickerLabelError").textContent = "";
  renderLabelOptions();
  element("labelPicker").showModal();
  positionLabelPicker();
  element("confirmPickedLabel").focus({preventScroll:true});
  draw();
}

function cancelLabelPicker() {
  if (state.labelPickerSaving) return;
  state.pendingShape = null;
  element("labelPicker").close();
  draw();
}

async function applyPickedLabel(name) {
  if (!state.pendingShape || state.labelPickerSaving || state.busy) return;
  const label = (name ?? element("pickerLabelInput").value).trim() || state.project.classes[state.activeClass].name;
  const pending = state.pendingShape;
  state.labelPickerSaving = true;
  element("pickerLabelError").textContent = "";
  element("labelPicker").querySelectorAll("button,input").forEach(node => {node.disabled = true;});
  try {
    if (!state.project.classes.some(item => item.name === label)) {
      const project = await api(projectUrl("/labels"), "POST", {name:label});
      Object.assign(state.project, project);
      element("projectSubtitle").textContent = `${state.project.batches.length} 个批次 · ${state.project.classes.length} 个标签 · 原始数据只读`;
    }
    state.activeClass = state.project.classes.findIndex(item => item.name === label);
    if (state.activeClass < 0) throw new Error("标签未创建成功，请重新选择");
    state.pendingShape = null;
    addShape(pending.points, "rectangle");
    renderClasses();
    setTool(efficiency.boxTool());
    element("labelPicker").close();
    canvas.focus({preventScroll:true});
  } catch (error) {
    element("pickerLabelError").textContent = error.message;
    positionLabelPicker();
  } finally {
    state.labelPickerSaving = false;
    element("labelPicker").querySelectorAll("button,input").forEach(node => {node.disabled = false;});
    if (element("labelPicker").open) element("pickerLabelInput").focus({preventScroll:true});
  }
}

function deleteShape() {
  if (!state.current || state.selected < 0 || state.busy) return;
  checkpoint();
  state.current.document.shapes.splice(state.selected,1);
  state.selected = -1;
  changed();
}

function setTool(tool) {
  state.tool = tool;
  state.polygon = [];
  state.rectangle = null;
  state.drag = null;
  document.querySelectorAll("[data-tool]").forEach(button => button.classList.toggle("active",button.dataset.tool === tool));
  canvas.style.cursor = tool === "pan" ? "grab" : tool === "select" ? "default" : "crosshair";
  efficiency.toolChanged(tool);
  draw();
}

function viewport() { const rect = element("canvasWrap").getBoundingClientRect(); return {width:rect.width,height:rect.height}; }
function toggleFocusMode(force) {
  const focused = typeof force === "boolean" ? force : !document.body.classList.contains("focus-mode");
  document.body.classList.toggle("focus-mode", focused);
  element("focusButton").textContent = focused ? "⛶ 恢复工作区" : "⛶ 大画布";
  element("focusButton").setAttribute("aria-pressed", String(focused));
  localStorage.setItem("labelpilot-focus", String(focused));
  requestAnimationFrame(resizeCanvas);
}
function fitImage() {
  if (!state.image) return draw();
  const size = viewport();
  state.scale = Math.min((size.width-16)/state.current.width,(size.height-22)/state.current.height);
  state.panX = (size.width-state.current.width*state.scale)/2;
  state.panY = (size.height-state.current.height*state.scale)/2;
  draw();
}
function resizeCanvas() {
  const size = viewport();
  canvas.width = Math.round(size.width * devicePixelRatio);
  canvas.height = Math.round(size.height * devicePixelRatio);
  if (!state.drag && !efficiency.refocus()) fitImage();
}
function screenPoint(point) { return [point[0]*state.scale+state.panX,point[1]*state.scale+state.panY]; }
function updateCrosshair(event) {
  if (event) state.canvasPointer = {clientX:event.clientX,clientY:event.clientY};
  const guide = element("canvasCrosshair");
  const pointer = state.canvasPointer;
  const rect = canvas.getBoundingClientRect();
  const horizontal = pointer ? pointer.clientX-rect.left : -1;
  const vertical = pointer ? pointer.clientY-rect.top : -1;
  guide.hidden = !pointer || !state.current || !state.image || state.tool!=="rectangle" || state.busy ||
    state.space || state.drag?.type==="pan" || !!document.querySelector("dialog[open]") ||
    horizontal<0 || vertical<0 || horizontal>=rect.width || vertical>=rect.height;
  if (!guide.hidden) {
    guide.style.setProperty("--crosshair-x",`${horizontal}px`);
    guide.style.setProperty("--crosshair-y",`${vertical}px`);
  }
}
function imagePoint(event,clamp=false) {
  const rect = canvas.getBoundingClientRect();
  let horizontal = (event.clientX-rect.left-state.panX)/state.scale;
  let vertical = (event.clientY-rect.top-state.panY)/state.scale;
  if (clamp && state.current) { horizontal=Math.max(0,Math.min(state.current.width,horizontal)); vertical=Math.max(0,Math.min(state.current.height,vertical)); }
  return [horizontal,vertical];
}
function bounds(shape) {
  const horizontal = shape.points.map(point => point[0]);
  const vertical = shape.points.map(point => point[1]);
  return [Math.min(...horizontal),Math.min(...vertical),Math.max(...horizontal),Math.max(...vertical)];
}
function rectanglePoints(left,top,right,bottom) { return [[left,top],[right,top],[right,bottom],[left,bottom]]; }
function shapePoints(shape) { return shape.shape_type === "rectangle" ? rectanglePoints(...bounds(shape)) : shape.points; }
function handles(shape) {
  if (shape.shape_type !== "rectangle") return shape.points;
  const [left,top,right,bottom] = bounds(shape);
  return [[left,top],[(left+right)/2,top],[right,top],[right,(top+bottom)/2],[right,bottom],[(left+right)/2,bottom],[left,bottom],[left,(top+bottom)/2]];
}
function pathPoints(points,close=true) {
  context.beginPath();
  points.forEach((point,index) => {const position=screenPoint(point);if(index===0)context.moveTo(...position);else context.lineTo(...position);});
  if (close) context.closePath();
}
function draw() {
  updateCrosshair();
  const size = viewport();
  context.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  context.clearRect(0,0,size.width,size.height);
  if (!state.image || !state.current) return;
  context.save();
  context.shadowColor = "#0009"; context.shadowBlur = 18;
  context.drawImage(state.image,state.panX,state.panY,state.current.width*state.scale,state.current.height*state.scale);
  context.restore();
  state.current.document.shapes.forEach((shape,index) => {
    const info = classInfo(shape.label);
    const selected = index === state.selected;
    pathPoints(shapePoints(shape));
    context.fillStyle = info.color+(selected ? "25" : "0b"); context.fill();
    context.strokeStyle=info.color; context.lineWidth=selected?2:1.2; context.stroke();
    const [left,top] = bounds(shape);
    if (state.showLabels || selected) {
      const position=screenPoint([left,top]);
      const label=info.name+(shape.score == null?"":` ${Math.round(shape.score*100)}%`);
      context.font="10px 'Segoe UI','Microsoft YaHei',sans-serif";
      const textWidth=context.measureText(label).width+9;
      const textY=Math.max(0,position[1]-17);
      context.fillStyle=info.color; context.fillRect(position[0]-.5,textY,textWidth,16);
      context.fillStyle="#10131a"; context.fillText(label,position[0]+4,textY+11.5);
    }
    if (selected) for (const point of handles(shape)) {
      const position=screenPoint(point);
      context.fillStyle="#f5f1ff"; context.strokeStyle=info.color; context.lineWidth=1.2;
      context.fillRect(position[0]-3,position[1]-3,6,6); context.strokeRect(position[0]-3,position[1]-3,6,6);
    }
  });
  if (state.rectangle || state.pendingShape) {
    const rectangle = state.rectangle;
    const points = rectangle ? rectanglePoints(Math.min(rectangle.start[0],rectangle.end[0]),Math.min(rectangle.start[1],rectangle.end[1]),Math.max(rectangle.start[0],rectangle.end[0]),Math.max(rectangle.start[1],rectangle.end[1])) : state.pendingShape.points;
    pathPoints(points);
    context.strokeStyle=state.project.classes[state.activeClass].color; context.lineWidth=1.5; context.setLineDash([5,3]); context.stroke(); context.setLineDash([]);
  }
  if (state.polygon.length) {
    const color=state.project.classes[state.activeClass].color;
    pathPoints(state.polygon,false); context.strokeStyle=color; context.lineWidth=1.5; context.stroke();
    for(const point of state.polygon){const position=screenPoint(point);context.fillStyle=color;context.beginPath();context.arc(...position,3,0,Math.PI*2);context.fill();}
  }
  efficiency.drawExtras();
  workspaceResume.observe();
  element("fitButton").textContent=`${Math.round(state.scale*100)}%`;
}
function zoom(factor,position) {
  if (!state.image) return;
  const size=viewport();
  const center=position || [size.width/2,size.height/2];
  const next=Math.max(.03,Math.min(16,state.scale*factor));
  state.panX=center[0]-(center[0]-state.panX)*next/state.scale;
  state.panY=center[1]-(center[1]-state.panY)*next/state.scale;
  state.scale=next; draw();
}
function pointInside(point,polygon) {
  let inside=false;
  for(let index=0,previous=polygon.length-1;index<polygon.length;previous=index++) {
    const currentPoint=polygon[index], previousPoint=polygon[previous];
    if((currentPoint[1]>point[1]) !== (previousPoint[1]>point[1]) && point[0]<(previousPoint[0]-currentPoint[0])*(point[1]-currentPoint[1])/(previousPoint[1]-currentPoint[1])+currentPoint[0]) inside=!inside;
  }
  return inside;
}
function addShape(points,kind) {
  checkpoint();
  state.current.document.shapes.push({label:state.project.classes[state.activeClass].name,points,shape_type:kind,
    score:null,group_id:null,description:"",flags:{},attributes:{},difficult:false});
  state.selected=state.current.document.shapes.length-1;
  changed();
}
function finishPolygon() {
  if(state.polygon.length<3)return toast("多边形至少需要三个顶点");
  const points=copy(state.polygon);
  if(points.length>3 && Math.hypot(points.at(-1)[0]-points.at(-2)[0],points.at(-1)[1]-points.at(-2)[1])*state.scale<3)points.pop();
  state.polygon=[];
  addShape(points,"polygon");
  setTool("select");
}

canvas.addEventListener("pointerdown",event => {
  if(!state.current || !state.image || state.busy || state.pendingShape)return;
  event.preventDefault();
  canvas.focus({preventScroll:true});
  updateCrosshair(event);
  const point=imagePoint(event,true);
  const actual=imagePoint(event);
  if(state.space || state.tool==="pan" || event.button===1 || event.button===2){canvas.setPointerCapture(event.pointerId);state.drag={type:"pan",client:[event.clientX,event.clientY],pan:[state.panX,state.panY]};updateCrosshair();return;}
  if(event.button!==0)return;
  if(state.tool==="rectangle" && state.rectangle){
    const start=state.rectangle.start;
    state.rectangle.end=point;
    if(Math.abs(point[0]-start[0])*state.scale<=3 || Math.abs(point[1]-start[1])*state.scale<=3){draw();return;}
    state.rectangle=null;
    openLabelPicker(rectanglePoints(Math.min(start[0],point[0]),Math.min(start[1],point[1]),Math.max(start[0],point[0]),Math.max(start[1],point[1])),point);
    return;
  }
  if(actual[0]<0 || actual[1]<0 || actual[0]>state.current.width || actual[1]>state.current.height)return;
  if(efficiency.handleCanvas(point))return;
  if(state.tool==="rectangle"){state.rectangle={start:point,end:point};state.selected=-1;renderObjects();draw();return;}
  if(state.tool==="polygon"){state.polygon.push(point);draw();return;}
  canvas.setPointerCapture(event.pointerId);
  const selected=state.current.document.shapes[state.selected];
  if(selected){
    const handle=handles(selected).findIndex(position => Math.hypot(position[0]-point[0],position[1]-point[1])*state.scale<8);
    if(handle>=0){state.drag={type:"resize",handle,start:point,original:copy(selected),before:snapshot()};return;}
  }
  const matches=state.current.document.shapes.map((shape,index) => ({shape,index})).filter(entry=>pointInside(point,shapePoints(entry.shape)));
  matches.sort((first,second)=>{const firstBounds=bounds(first.shape),secondBounds=bounds(second.shape);return (firstBounds[2]-firstBounds[0])*(firstBounds[3]-firstBounds[1])-(secondBounds[2]-secondBounds[0])*(secondBounds[3]-secondBounds[1]);});
  selectShape(matches[0]?.index ?? -1);
  if(state.selected>=0)state.drag={type:"move",start:point,original:copy(state.current.document.shapes[state.selected]),before:snapshot()};
});
canvas.addEventListener("pointermove",event => {
  updateCrosshair(event);
  if(!state.current)return;
  const point=imagePoint(event,true);
  element("coordinateText").textContent=`x ${Math.round(point[0])} · y ${Math.round(point[1])}`;
  if(state.rectangle && !state.busy){state.rectangle.end=point;draw();}
  const drag=state.drag;
  if(!drag || state.busy)return;
  if(drag.type==="pan"){state.panX=drag.pan[0]+event.clientX-drag.client[0];state.panY=drag.pan[1]+event.clientY-drag.client[1];}
  if(drag.type==="move"){
    const originalBounds=bounds(drag.original);
    const horizontal=Math.max(-originalBounds[0],Math.min(state.current.width-originalBounds[2],point[0]-drag.start[0]));
    const vertical=Math.max(-originalBounds[1],Math.min(state.current.height-originalBounds[3],point[1]-drag.start[1]));
    state.current.document.shapes[state.selected].points=drag.original.points.map(position=>[position[0]+horizontal,position[1]+vertical]);
  }
  if(drag.type==="resize"){
    const shape=state.current.document.shapes[state.selected];
    if(shape.shape_type==="polygon")shape.points[drag.handle]=point;
    else{
      let [left,top,right,bottom]=bounds(drag.original);
      if([0,6,7].includes(drag.handle))left=Math.min(point[0],right-1);
      if([2,3,4].includes(drag.handle))right=Math.max(point[0],left+1);
      if([0,1,2].includes(drag.handle))top=Math.min(point[1],bottom-1);
      if([4,5,6].includes(drag.handle))bottom=Math.max(point[1],top+1);
      shape.points=rectanglePoints(left,top,right,bottom);
    }
  }
  draw();
});
function finishDrag() {
  const drag=state.drag;
  if(!drag)return;
  state.drag=null;
  if(drag.before && JSON.stringify(drag.before)!==JSON.stringify(state.current.document)){checkpoint(drag.before);changed();}
  draw();
}
canvas.addEventListener("pointerup",finishDrag);
canvas.addEventListener("pointerenter",updateCrosshair);
canvas.addEventListener("pointerleave",()=>{state.canvasPointer=null;updateCrosshair();});
canvas.addEventListener("pointercancel",()=>{if(state.drag?.before)state.current.document=state.drag.before;state.drag=null;state.rectangle=null;state.canvasPointer=null;draw();});
canvas.addEventListener("dblclick",()=>{if(state.tool==="polygon" && !state.busy)finishPolygon();});
canvas.addEventListener("contextmenu",event=>event.preventDefault());
canvas.addEventListener("wheel",event=>{event.preventDefault();if(state.busy)return;const rect=canvas.getBoundingClientRect();zoom(Math.exp(-event.deltaY*.001),[event.clientX-rect.left,event.clientY-rect.top]);},{passive:false});
new ResizeObserver(resizeCanvas).observe(element("canvasWrap"));

function modelOptions() {
  return {confidence:Number(element("confidence").value),iou:Number(element("iou").value),image_size:Number(element("imageSize").value)};
}
async function saveModelConfig() {
  const path=element("modelPath").value.trim();
  if(path===state.project.model_path)return;
  const project=await api(projectUrl("/model"),"POST",{path});
  Object.assign(state.project,project);
  element("modelSelectionStatus").textContent="模型配置已保存，下次预标注将使用此权重；首次推理校验类别。";
  toast("模型路径已保存；首次推理会校验类别映射");
}
function modelSelectionChanged() {
  const path=element("modelPath").value.trim();
  element("modelPath").title=path;
  element("modelSelectionStatus").textContent=path===state.project?.model_path ? "当前已保存的模型" : path ? "已选择新模型，尚未保存配置" : "请选择可信的本地 .pt 模型";
}
function renderModelBrowser() {
  const listing=modelBrowser.listing;
  if(!listing)return;
  element("modelBrowserPath").value=listing.path;
  element("modelBrowserPath").title=listing.path;
  element("modelBrowserDrive").innerHTML='<option value="">磁盘位置</option>'+listing.roots.map(path=>`<option value="${escapeHtml(path)}">${escapeHtml(path)}</option>`).join("");
  element("modelBrowserCount").textContent=`${listing.search ? "含子文件夹 · " : ""}${listing.directories.length} 个文件夹 · ${listing.files.length} 个模型`;
  const folders=listing.directories.map(item=>`<button type="button" class="model-file-row folder" data-model-directory="${escapeHtml(item.path)}" title="${escapeHtml(item.path)}"><b>▸</b><span>${escapeHtml(item.name)}</span><small>打开文件夹</small></button>`).join("");
  const files=listing.files.map(item=>`<button type="button" class="model-file-row" data-model-file="${escapeHtml(item.path)}" title="${escapeHtml(item.path)}"><b>.pt</b><span>${escapeHtml(item.name)}</span><small>${(item.size/1024/1024).toFixed(1)} MB · 选择</small></button>`).join("");
  element("modelBrowserList").innerHTML=folders+files || `<p class="empty-list">${listing.search ? "未找到 .pt 模型，请选择其他文件夹。" : "此处没有子文件夹或 .pt 模型。"}</p>`;
  element("modelBrowserError").textContent=listing.partial ? "结果可能不完整（访问权限、扫描时间或数量上限）；请选择更具体的文件夹继续查找。" : "";
}
function setModelBrowserLoading(value) {
  modelBrowser.loading=value;
  for(const id of ["modelBrowserDrive","modelBrowserPath","modelBrowserGo","searchModelFolder"])element(id).disabled=value;
  element("modelBrowserUp").disabled=value || !modelBrowser.listing?.parent;
  element("searchModelFolder").disabled=value || !modelBrowser.listing;
  element("modelBrowserList").querySelectorAll("button").forEach(button=>{button.disabled=value;});
}
async function loadModelDirectory(path, search=false) {
  const sequence=++modelBrowser.sequence;
  const projectId=state.project.id;
  setModelBrowserLoading(true);
  element("modelBrowserError").textContent="";
  element("modelBrowserCount").textContent=search ? "正在查找 .pt 模型（包含子文件夹）…" : "正在读取本机文件夹…";
  try {
    const listing=await api(projectUrl(search ? "/model/search" : "/model/browse"),"POST",{path});
    if(sequence!==modelBrowser.sequence || !element("modelBrowserDialog").open || state.project.id!==projectId)return;
    modelBrowser.listing=listing;
    renderModelBrowser();
  } catch(error) {
    if(sequence!==modelBrowser.sequence)return;
    element("modelBrowserError").textContent=error.message;
    element("modelBrowserCount").textContent="未能读取，请修改路径或返回上一级。";
  } finally {if(sequence===modelBrowser.sequence)setModelBrowserLoading(false);}
}
async function openModelBrowser(mode, target="modelPath") {
  if(!state.project || state.busy)return;
  modelBrowser.mode=mode;
  modelBrowser.target=target;
  modelBrowser.listing=null;
  element("modelBrowserList").innerHTML="";
  element("modelBrowserDrive").innerHTML='<option value="">磁盘位置</option>';
  element("modelBrowserPath").value="";
  element("modelBrowserTitle").textContent=mode==="folder" ? "选择模型文件夹" : "选择本机模型文件";
  element("modelBrowserHelp").textContent=mode==="folder" ? "打开目标文件夹，点击下方“选择此文件夹并查找模型”，再从结果中明确选择一个 .pt 文件。" : "点击文件夹进入，点击 .pt 文件选择；仅读取路径，不上传或加载文件。";
  element("searchModelFolder").hidden=mode!=="folder";
  element("modelBrowserDialog").showModal();
  await loadModelDirectory(null);
}
function chooseModelFile(path) {
  if(modelBrowser.loading)return;
  const field=element(modelBrowser.target);
  field.value=path;
  if(modelBrowser.target==="modelPath")modelSelectionChanged();
  element("modelBrowserDialog").close();
  field.focus({preventScroll:true});
}
function updateImageSelection() {
  const files=projectImages.files;
  element("newImageRoot").title=element("newImageRoot").value;
  element("newImageSelection").textContent=files.length ? `仅导入已选 ${files.length} 张图片：${files.slice(0,3).join("、")}${files.length>3 ? " 等" : ""}。重新扫描不会加入其他图片。` : "文件夹模式：读取目录及子目录内的图片；文件模式：仅导入明确选中的图片。";
  element("useEntireImageFolder").hidden=!files.length;
}
function updateImageBrowserControls() {
  const {listing,loading,valid,selected,mode}=imageBrowser;
  for(const id of ["imageBrowserDrive","imageBrowserPath","imageBrowserGo"])element(id).disabled=loading;
  element("imageBrowserUp").disabled=loading || !listing?.parent;
  element("confirmImageSelection").disabled=loading || !valid || (mode==="file" && !selected.size);
  element("confirmImageSelection").textContent=mode==="folder" ? "选择此文件夹" : `使用所选图片（${selected.size} 张）`;
  element("imageBrowserList").querySelectorAll("button").forEach(button=>{button.disabled=loading || !valid || (mode==="folder" && button.hasAttribute("data-image-file"));});
  if(valid)element("imageBrowserCount").textContent=`${listing.directories.length} 个文件夹 · ${listing.files.length} 张图片${mode==="file" ? ` · 已选 ${selected.size} 张` : ""}`;
}
function renderImageBrowser() {
  const listing=imageBrowser.listing;
  element("imageBrowserPath").value=listing.path;
  element("imageBrowserPath").title=listing.path;
  element("imageBrowserDrive").innerHTML='<option value="">磁盘位置</option>'+listing.roots.map(path=>`<option value="${escapeHtml(path)}">${escapeHtml(path)}</option>`).join("");
  const folders=listing.directories.map(item=>`<button type="button" class="model-file-row folder" data-image-directory="${escapeHtml(item.path)}" title="${escapeHtml(item.path)}"><b>▸</b><span>${escapeHtml(item.name)}</span><small>打开文件夹</small></button>`).join("");
  const files=listing.files.map(item=>`<button type="button" class="model-file-row image-file-row" data-image-file="${escapeHtml(item.path)}" aria-pressed="false" title="${escapeHtml(item.path)}"><b>图片</b><span>${escapeHtml(item.name)}</span><small>${(item.size/1024/1024).toFixed(2)} MB${imageBrowser.mode==="file" ? " · 点击勾选" : ""}</small></button>`).join("");
  element("imageBrowserList").innerHTML=folders+files || '<p class="empty-list">此处没有子文件夹或支持的图片。</p>';
  element("imageBrowserError").textContent=listing.partial ? "当前列表可能不完整（权限或数量上限）；可进入更具体的文件夹继续选择。" : "";
}
async function loadImageDirectory(path) {
  const sequence=++imageBrowser.sequence;
  imageBrowser.loading=true;imageBrowser.valid=false;
  element("imageBrowserError").textContent="";
  element("imageBrowserCount").textContent="正在读取本机图片文件夹…";
  updateImageBrowserControls();
  try {
    const listing=await api("/api/images/browse","POST",{path});
    if(sequence!==imageBrowser.sequence || !element("imageBrowserDialog").open)return;
    imageBrowser.listing=listing;imageBrowser.selected.clear();imageBrowser.valid=true;
    renderImageBrowser();
  } catch(error) {
    if(sequence!==imageBrowser.sequence)return;
    element("imageBrowserError").textContent=error.message;
    element("imageBrowserCount").textContent="未能读取，请修改路径或返回上一级；尚未改变所选图片。";
  } finally {if(sequence===imageBrowser.sequence){imageBrowser.loading=false;updateImageBrowserControls();}}
}
async function openImageBrowser(mode, target="newImageRoot") {
  if(state.busy)return;
  imageBrowser.mode=mode;imageBrowser.target=target;imageBrowser.listing=null;imageBrowser.selected.clear();imageBrowser.valid=false;
  element("imageBrowserTitle").textContent=mode==="folder" ? "选择待标注图片文件夹" : "选择待标注图片（可多选）";
  element("imageBrowserHelp").textContent=mode==="folder" ? "进入目标目录后点击“选择此文件夹”，导入其中及子目录内的图片。仅引用路径，不上传或修改源文件。" : "点击图片勾选或取消，可选择同一文件夹内的多张图片；切换文件夹会清空本次勾选。支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF。";
  if (target !== "newImageRoot") element("imageBrowserHelp").textContent = "选择当前项目的原图目录或子目录作为收图范围。";
  element("imageBrowserList").innerHTML="";
  element("imageBrowserDrive").innerHTML='<option value="">磁盘位置</option>';
  element("imageBrowserPath").value="";
  element("imageBrowserDialog").showModal();
  const field = element(target).value.trim();
  await loadImageDirectory(target === "newImageRoot" ? field || state.project?.image_root || null : state.project.image_root);
}
function confirmImageSelection() {
  if(imageBrowser.loading || !imageBrowser.valid || (imageBrowser.mode==="file" && !imageBrowser.selected.size))return;
  const listing=imageBrowser.listing;
  const field = element(imageBrowser.target);
  field.value=listing.path;
  if (imageBrowser.target === "newImageRoot") {
    projectImages.files=imageBrowser.mode==="file" ? listing.files.filter(item=>imageBrowser.selected.has(item.path)).map(item=>item.name) : [];
    updateImageSelection();
  }
  element("imageBrowserDialog").close();
  field.focus({preventScroll:true});
}
function openModelDialog() {
  if(!state.project)return;
  element("modelPath").value=state.project.model_path;
  modelSelectionChanged();
  const scope = filters().status === "curated" ? "相似筛图后保留的未开始图片" : "未开始图片";
  element("batchScope").textContent=`${filters().batch || "全部批次"}${filters().query ? " · 搜索 "+filters().query : ""} · ${scope}`;
  element("modelDialog").showModal();
  pollJobs().catch(error=>toast(error.message,true));
}
async function predictCurrent() {
  if(!state.current || state.busy)return;
  if(!element("modelPath").value.trim()){openModelDialog();return;}
  await saveModelConfig();
  element("modelDialog").close();
  setBusy(true,"模型正在推理；首次加载可能需要片刻…");
  toast("正在运行本地模型，原有标注不会自动替换");
  try{
    state.prediction=await api(itemUrl("/predict"),"POST",modelOptions());
    const prediction=state.prediction;
    const score=Number(element("confidence").value).toFixed(2);
    element("predictionSummary").textContent=`发现 ${prediction.shapes.length} 个候选对象 · 阈值 ${score} · ${prediction.provenance.device}。当前画布有 ${state.current.document.shapes.length} 个对象。请选择追加或替换，应用后仍需保存和人工确认。`;
    element("predictionDialog").showModal();
  }finally{setBusy(false);}
}
function applyPrediction(append) {
  if(!state.prediction || !state.current)return;
  if(state.prediction.base_version!==state.current.version || state.prediction.image_sha256!==state.current.image_sha256)throw new Error("标注版本或图片已变化，请重新推理");
  checkpoint();
  state.current.document.shapes=append ? state.current.document.shapes.concat(copy(state.prediction.shapes)) : copy(state.prediction.shapes);
  state.current.document.labelpilot={...(state.current.document.labelpilot || {}),prediction:copy(state.prediction.provenance)};
  state.selected=-1;
  state.prediction=null;
  element("predictionDialog").close();
  changed();
  toast("模型建议已应用，尚未保存 / 审核"+(append?"；追加可能存在重叠框，请检查":""));
}
async function startBatch() {
  if(state.busy)return;
  await saveModelConfig();
  if(!await ensureSaved())return;
  const limit=Number(element("batchLimit").value);
  const curated = filters().status === "curated";
  const scope = curated ? "筛后队列中保留的" : "";
  if(!window.confirm(`即将对 ${filters().batch || "全部批次"}内最多 ${limit} 张${scope}未开始图片运行本地模型，结果将进入待审核队列。继续？`))return;
  await api(projectUrl("/jobs"),"POST",{...modelOptions(),batch:filters().batch,query:filters().query,limit,curated});
  toast("批量任务已启动，可以继续标注其他图片");
  await pollJobs();
}
async function pollJobs() {
  if(!state.project)return;
  const project = state.project.id;
  const jobs=await api(projectUrl("/jobs"));
  if (state.project?.id !== project) return;
  efficiency.updateJobs(jobs);
  const jobNames={queued:"等待中",running:"推理中",completed:"已完成",completed_with_errors:"完成，含失败",cancelled:"已取消",interrupted:"启动前任务已中断",failed:"失败"};
  const running=jobs.find(job=>["running","queued"].includes(job.status));
  element("jobStatus").hidden=!running;
  const jobStages = {audit:"疑点扫描", routing:"智能审核分流"};
  if(running)element("jobStatus").textContent=`✦ ${jobStages[running.kind] || "批量预标注"} ${running.done+running.skipped+running.failed} / ${running.total}`;
  element("jobsList").innerHTML=jobs.slice(0,4).map(job=>`<div class="job-row"><div><strong>${jobStages[job.kind] ? jobStages[job.kind] + " · " : ""}${jobNames[job.status] || escapeHtml(job.status)} · ${job.done} / ${job.total}</strong><small>跳过 ${job.skipped} · 失败 ${job.failed}${job.errors?.length ? " · "+escapeHtml(job.errors.at(-1).error) : ""}</small></div>${["running","queued"].includes(job.status) ? `<button class="button" data-cancel-job="${job.id}">取消</button>`:""}</div>`).join("");
  const signature=jobs.map(job=>`${job.id}:${job.status}:${job.done}`).join("|");
  if(state.jobSignature && state.jobSignature!==signature && !state.busy)await refreshList();
  state.jobSignature=signature;
}
async function showHistory() {
  if(!state.current)return;
  const revisions=await api(itemUrl("/history"));
  element("historyList").innerHTML=revisions.map(revision=>`<div class="history-row"><div><strong>版本 ${revision.version} · ${escapeHtml(statusNames[revision.status])}</strong><small>${escapeHtml(new Date(revision.updated).toLocaleString())}</small></div><button class="button" data-restore="${revision.version}">恢复为新草稿</button></div>`).join("") || '<p class="muted">尚未在本工具保存。首次保存会同时保留原始导入版本，后续每次保存都有独立版本。</p>';
  element("historyDialog").showModal();
}
async function restoreRevision(revision) {
  if(state.busy || !state.current)return;
  if(!window.confirm("恢复历史版本会作为一个新草稿保存。当前未保存修改将丢弃，已保存历史不会删除。继续？"))return;
  await api(itemUrl("/restore"),"POST",{revision,expected_version:state.current.version});
  const itemId=state.current.item.id;
  setDirty(false);
  element("historyDialog").close();
  await loadItem(itemId);
  await refreshList();
  toast("历史版本已恢复为新的草稿，需要重新人工确认");
}
function exportSplitOptions() {
  if(!element("splitDataset").checked)return null;
  const ratios={train:element("splitTrain").valueAsNumber,val:element("splitVal").valueAsNumber,test:element("splitTest").valueAsNumber};
  if(Object.values(ratios).some(value=>!Number.isFinite(value) || value<0 || value>100))throw new Error("训练 / 验证 / 测试比例必须为 0 到 100 的数字");
  const total=ratios.train+ratios.val+ratios.test;
  if(Math.abs(total-100)>1e-6 || ratios.train<=0)throw new Error(`三个比例之和必须为 100%，且训练集比例必须大于 0（当前合计 ${Number(total.toFixed(6))}%）`);
  const seed=element("splitSeed").valueAsNumber;
  if(!Number.isInteger(seed) || seed<0 || seed>4294967295)throw new Error("随机种子必须为 0 到 4294967295 的整数");
  return {...ratios, seed, unit:element("exportSplitUnit").value};
}

function setExportPreviewStatus(message, error=false) {
  const status = element("exportPreviewStatus");
  status.textContent = message;
  status.hidden = !message;
  status.dataset.error = String(error);
}

function exportAugmentationOptions() {
  const preset = exportAugmentation.presets.find(item => item.id === element("augmentationPreset").value);
  for (const id of ["augmentationCopies", "augmentationSeed"]) element(id).disabled = !preset;
  element("augmentationDescription").textContent = preset?.description || "不生成额外图片。";
  const count = preset?.preset_count || 1;
  element("augmentationCopiesLabel").textContent = count > 1 ? "每张训练图片、每种方案额外生成" : "每张训练图片额外生成";
  const summary = element("augmentationSummary");
  summary.textContent = "";
  summary.dataset.invalid = "false";
  if (!preset) return null;
  const copies = element("augmentationCopies").valueAsNumber;
  const seed = element("augmentationSeed").valueAsNumber;
  let issue;
  if (!element("splitDataset").checked) issue = "增强需先划分数据集。仅训练可设为 100 / 0 / 0。";
  else if (!element("includeImages").checked) issue = "数据增强必须勾选包含图片。";
  else if (!element("reviewedOnly").checked) issue = "数据增强仅支持已人工确认的标注。";
  else if (!Number.isInteger(copies) || copies < 1 || copies > exportAugmentation.maxCopies) issue = `额外生成数量必须为 1 到 ${exportAugmentation.maxCopies} 的整数。`;
  else if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) issue = "增强种子必须在 0 到 4294967295 之间。";
  if (issue) {
    summary.textContent = issue;
    summary.dataset.invalid = "true";
    throw new Error(issue);
  }
  const prefix = count > 1 ? `${count} 种方案分别生成，每方案 ${copies} 张。` : "";
  summary.textContent = `${prefix}每张训练图额外生成 ${copies * count} 张，含原图共 ${1 + copies * count} 张（${1 + copies * count} 倍）。`;
  return {preset:preset.id, copies, seed};
}

async function loadAugmentationCatalog() {
  if (exportAugmentation.loaded) return;
  try {
    const catalog = await api("/api/augmentation-presets");
    exportAugmentation.presets = catalog.presets;
    exportAugmentation.maxCopies = catalog.max_copies;
    exportAugmentation.loaded = true;
    element("augmentationPreset").append(...catalog.presets.map(item => new Option(item.name, item.id)));
    element("augmentationCopies").max = catalog.max_copies;
    element("augmentationPreset").disabled = false;
  } catch (error) {
    element("augmentationDescription").textContent = `增强方案加载失败。${error.message}`;
  }
}

function renderExportPreview(data, options) {
  const counts = data.split_counts;
  const configuredSplits = Object.keys(exportSplitLabels);
  const activeSplits = configuredSplits.filter(name => counts?.[name] > 0);
  const notes = [];
  if (data.excluded) notes.push(`${data.excluded} 张未纳入`);
  if (data.unreviewed) notes.push(`${data.unreviewed} 张待审核`);
  if (data.group_count) notes.push(`${data.group_count} 个分组`);
  if (data.augmented_images) notes.push(`训练集将额外生成 ${data.augmented_images} 张增强图`);
  if (state.dirty) notes.push("当前画布有未保存的修改，预检使用已保存版本");
  setExportPreviewStatus(notes.join(" · "));

  element("exportImageCount").textContent = data.images.toLocaleString();
  element("exportObjectCount").textContent = data.annotations.toLocaleString();
  element("exportEmptyCount").textContent = data.confirmed_empty.toLocaleString();
  element("exportSplitDistribution").hidden = !counts;
  if (counts) {
    element("exportSplitBar").innerHTML = configuredSplits.map(name =>
      `<span data-subset="${name}" style="width:${100 * counts[name] / data.images}%"></span>`).join("");
    element("exportSplitCounts").innerHTML = configuredSplits.map(name =>
      `<div><dt>${exportSplitLabels[name]}</dt><dd>${counts[name].toLocaleString()} 张</dd></div>`).join("");
  }

  const issues = [];
  if (data.blocking_issue) issues.push(data.blocking_issue);
  if (data.unreviewed_empty) issues.push(`${data.unreviewed_empty} 张待审核图没有标注对象，作为负样本使用前请先确认。`);
  if (data.cross_split_sources) issues.push(`${data.cross_split_sources} 个已知来源组跨越多个集合。`);
  if (activeSplits.length > 1 && data.missing_sources) {
    issues.push(`${data.missing_sources} 张图片缺少来源组，无法检查这些图片是否同源跨集。`);
  }
  if (counts) {
    const emptySplits = activeSplits.filter(name => data.class_counts.every(row => row.splits[name] === 0));
    if (emptySplits.length) issues.push(`${emptySplits.map(name => `${exportSplitLabels[name]}集`).join("、")}没有标注对象。`);
    const missingTrain = data.class_counts.filter(row => row.splits.train === 0).length;
    if (missingTrain && !emptySplits.includes("train")) issues.push(`${missingTrain} 个类别在训练集中没有标注。`);
    if (options.format === "yolo" && counts.val === 0) issues.push("YOLO 训练前需要另行配置验证数据。");
  }
  element("exportPreviewIssues").innerHTML = issues.map(issue => `<p>${escapeHtml(issue)}</p>`).join("");
  element("exportPreviewIssues").hidden = !issues.length;
  element("exportCleaning").hidden = !data.cleaning;
  if (data.cleaning) {
    const labels = {duplicate:"重复副本", conflict:"标注冲突", unreadable:"图片不可用", annotation:"标注异常"};
    const excluded = Object.entries(data.cleaning.counts).filter(([, count]) => count);
    element("exportCleaningSummary").textContent = excluded.length
      ? excluded.map(([reason, count]) => `${labels[reason]} ${count} 张`).join(" · ") : "清洗检查完成 · 排除 0 张";
    element("exportCleaningDetails").hidden = !data.cleaning.examples.length;
    element("exportCleaningExamples").innerHTML = data.cleaning.examples.map(row => {
      const reviewLink = ["conflict", "annotation"].includes(row.reason)
        ? ` <button class="button" type="button" data-clean-item="${escapeHtml(row.id)}">查看图片</button>` : "";
      return `<li><strong>${escapeHtml(row.photo_id)}</strong> · ${labels[row.reason]} · ${escapeHtml(row.detail)}${reviewLink}</li>`;
    }).join("");
  }

  element("exportClassColumns").innerHTML = '<th scope="col">类别</th><th scope="col">合计</th>' + activeSplits.map(name =>
    `<th scope="col">${exportSplitLabels[name]}</th>`).join("");
  element("exportClassRows").innerHTML = data.class_counts.map(row => {
    const cells = activeSplits.map(name =>
      `<td${row.splits[name] ? "" : " data-gap"}>${row.splits[name].toLocaleString()}</td>`).join("");
    return `<tr><th scope="row">${escapeHtml(row.name)}</th><td>${row.total.toLocaleString()}</td>${cells}</tr>`;
  }).join("");
  element("exportPreviewContent").hidden = false;
}

function updateExportPreview() {
  if (exportPreview.running || !element("exportDialog").open) return;
  clearTimeout(exportPreview.timer);
  const sequence = ++exportPreview.sequence;
  exportPreview.plan = null;
  element("createExport").disabled = true;
  element("exportPreviewContent").hidden = true;
  const splitSettings = element("exportSplitSettings");
  splitSettings.disabled = !element("splitDataset").checked;
  splitSettings.hidden = splitSettings.disabled;
  element("exportSplitHint").textContent = exportSplitHints[element("exportSplitUnit").value];
  const summary = element("exportSplitSummary");
  const clean = element("cleanDataset").checked;
  element("reviewedOnly").disabled = clean;
  if (clean) element("reviewedOnly").checked = true;
  element("createExport").textContent = clean ? "下载清洗版 ZIP" : "生成 ZIP 并下载";
  let split;
  try {
    split = exportSplitOptions();
  } catch (error) {
    summary.textContent = error.message;
    summary.hidden = false;
    setExportPreviewStatus("请调整划分设置后查看预检结果。");
    return;
  }
  summary.textContent = "";
  summary.hidden = true;

  let augmentation;
  try { augmentation = exportAugmentationOptions(); }
  catch (error) { setExportPreviewStatus(error.message, true); return; }

  const options = {format:element("exportFormat").value, batch:element("exportBatch").value,
    reviewed_only:element("reviewedOnly").checked, include_images:element("includeImages").checked, clean, split, augmentation};
  const url = projectUrl("/export");
  setExportPreviewStatus(clean ? "正在读取原图并检查重复和标注异常…" : "正在统计本次导出的标注…");
  exportPreview.timer = setTimeout(async () => {
    try {
      const data = await api(`${url}/preview`, "POST", options);
      if (sequence !== exportPreview.sequence || !element("exportDialog").open) return;
      renderExportPreview(data, options);
      exportPreview.plan = {url, options, data};
      element("createExport").disabled = Boolean(data.blocking_issue);
    } catch (error) {
      if (sequence === exportPreview.sequence && element("exportDialog").open) setExportPreviewStatus(error.message, true);
    }
  }, 150);
}

async function openExportDialog(clean=false) {
  if (!state.project) return;
  if (exportPreview.running) return toast("正在生成上一次的导出包");
  element("exportResult").textContent = "";
  element("cleanDataset").checked = clean;
  if (clean) {
    element("augmentationPreset").value = "none";
    element("exportFormat").value = "labelme";
    element("exportBatch").value = filters().batch;
    element("includeImages").checked = true;
    element("splitDataset").checked = false;
  }
  element("exportDialog").showModal();
  element("exportDialog").querySelector(".dialog-body").scrollTop = 0;
  updateExportPreview();
  await loadAugmentationCatalog();
}

const trainingStatuses = {preparing:"准备清洗版数据", running:"训练中", cancelling:"正在停止", cancelled:"已停止",
  interrupted:"已中断", completed:"已完成", failed:"训练失败"};

function trainingMessage(message, error=false) {
  const status = element("trainingMessage");
  status.textContent = message;
  status.dataset.error = String(error);
}

function renderTrainingRuns(runs) {
  const active = runs.some(run => ["preparing", "running", "cancelling"].includes(run.status));
  element("startTraining").disabled = active || trainingSubmitting;
  if (JSON.stringify(runs) === JSON.stringify(trainingRuns)) return;
  trainingRuns = runs;
  element("trainingRuns").innerHTML = runs.map(run => {
    const url = escapeHtml(projectUrl(`/trainings/${run.id}`));
    const counts = run.split_counts;
    const progress = run.progress;
    const epoch = progress.epoch ?? 0;
    const epochs = progress.epochs ?? run.options.epochs;
    const metrics = [["mAP50–95", progress.metrics?.map50_95], ["mAP50", progress.metrics?.map50],
      ["召回率", progress.metrics?.recall], ["精确率", progress.metrics?.precision]];
    const scored = metrics.filter(([, value]) => Number.isFinite(value));
    const information = [escapeHtml(run.options.batch || "全部批次")];
    if (counts) information.push(`训练 ${counts.train} 张 · 验证 ${counts.val} 张`, `清洗排除 ${run.cleaned_out} 张`);
    if (run.blind_out) information.push(`盲测预留 ${run.blind_out} 张`);
    if (progress.device) information.push(escapeHtml(progress.device));
    const statusLabel = trainingStatuses[run.status];
    return `<article class="training-run" data-training-id="${escapeHtml(run.id)}">
      <div class="training-run-heading"><strong>${statusLabel}</strong><time>${new Date(run.created_at).toLocaleString()}</time></div>
      <p class="muted">${information.join(" · ")}</p>
      ${progress.epochs ? `<progress max="${epochs}" value="${epoch}" aria-label="训练轮数"></progress><small>${epoch} / ${epochs} 轮</small>` : ""}
      ${scored.length ? `<p>${scored.map(([name, value]) => `${name} ${value.toFixed(3)}`).join(" · ")}</p>` : ""}
      ${run.error ? `<p class="training-error">${escapeHtml(run.error)}</p>` : ""}
      <div class="training-run-actions">
        ${run.has_log ? `<a class="button" href="${url}/log" target="_blank" rel="noopener">日志</a>` : ""}
        ${run.has_dataset ? `<a class="button" href="${url}/dataset" download>数据清单</a>` : ""}
        ${run.weights ? `<a class="button" href="${url}/weights" download>下载权重</a><button class="button" data-use-training="${escapeHtml(run.id)}">设为预标注模型</button>` : ""}
        ${["preparing", "running"].includes(run.status) ? `<button class="button" data-cancel-training="${escapeHtml(run.id)}">停止训练</button>` : ""}
      </div>
    </article>`;
  }).join("") || '<p class="muted">当前项目暂无训练记录</p>';
}

async function refreshTrainingRuns() {
  if (!element("trainingDialog").open) return;
  const sequence = ++trainingSequence;
  const runs = await api(projectUrl("/trainings"));
  if (element("trainingDialog").open && sequence === trainingSequence) renderTrainingRuns(runs);
}

async function openTrainingDialog() {
  if (!state.project) return;
  element("trainingBatch").innerHTML = element("exportBatch").innerHTML;
  element("trainingBatch").value = filters().batch;
  element("trainingUnit").value = state.project.has_source_groups ? "source_group" : "image";
  element("trainingMode").value = state.project.model_path ? "fine_tune" : "scratch";
  element("trainingRuns").textContent = "正在读取训练记录…";
  trainingRuns = null;
  trainingMessage("");
  element("trainingRuntime").textContent = "正在检查本地训练环境…";
  element("trainingDialog").showModal();
  clearInterval(trainingTimer);
  trainingTimer = setInterval(() => refreshTrainingRuns().catch(error => trainingMessage(error.message, true)), 1500);
  await Promise.all([
    refreshTrainingRuns(),
    api("/api/system").then(system => {
      element("trainingRuntime").textContent = system.training_available ? `可用设备 ${system.device}${system.cuda ? " 或 CPU" : ""}`
        : "当前 Python 需要安装 PyTorch 和 Ultralytics。";
    })
  ]);
}

async function startTraining(event) {
  event.preventDefault();
  if (trainingSubmitting || element("startTraining").disabled) return;
  trainingSubmitting = true;
  element("startTraining").disabled = true;
  trainingMessage("正在创建训练任务…");
  const options = {batch:element("trainingBatch").value, unit:element("trainingUnit").value,
    mode:element("trainingMode").value, device:element("trainingDevice").value,
    epochs:element("trainingEpochs").valueAsNumber, image_size:element("trainingImageSize").valueAsNumber,
    batch_size:element("trainingBatchSize").valueAsNumber, seed:element("trainingSeed").valueAsNumber};
  try {
    await api(projectUrl("/trainings"), "POST", options);
    trainingMessage("训练任务已启动，可返回标注工作台。");
  } catch (error) {
    trainingMessage(error.message, true);
  } finally {
    trainingSubmitting = false;
    await refreshTrainingRuns();
  }
}

const comparisonStatuses = {preparing:"准备文件", running:"正在对比", cancelling:"正在停止", completed:"对比完成",
  cancelled:"已停止", interrupted:"服务关闭，任务已中断", failed:"对比失败"};
const comparisonActive = new Set(["preparing", "running", "cancelling"]);
const recallFormat = new Intl.NumberFormat("zh-CN", {style:"percent", maximumFractionDigits:1});
const comparisonReviewLabels = {old_missed:"旧版漏检", new_missed:"新版漏检", old_false_positives:"旧版误检",
  new_false_positives:"新版误检", newly_missed:"新版新增漏检", fixed_missed:"新版修复漏检"};

function comparisonMessage(id, message, error=false) {
  const node = element(id);
  if (node.textContent !== message) node.textContent = message;
  node.dataset.error = String(error);
}

function syncComparisonSet() {
  const selected = comparisonUi.sets.find(set => set.id === element("comparisonSet").value);
  element("comparisonSetInfo").textContent = selected
    ? `${selected.images} 张图 · ${selected.objects} 个真值框 · ${selected.confirmed_empty} 张已确认空图`
    : "先冻结一批已审核图片，供两版模型共用。";
  element("startComparison").disabled = !selected || comparisonUi.submitting ||
    comparisonUi.runs.some(run => comparisonActive.has(run.status));
}

function renderBlindsets(preferred) {
  const select = element("comparisonSet");
  const candidate = preferred || select.value;
  select.replaceChildren(...comparisonUi.sets.map(set =>
    new Option(`${set.name} · ${set.batch || "全部批次"} · ${new Date(set.created_at).toLocaleDateString()}`, set.id)));
  if (!select.options.length) select.add(new Option("暂无盲测集", ""));
  if (comparisonUi.sets.some(set => set.id === candidate)) select.value = candidate;
  syncComparisonSet();
}

function renderComparisonReport(run, report) {
  const {snapshot, models, options, classes} = report;
  element("comparisonReportMeta").textContent =
    `${snapshot.name} · ${snapshot.images} 张图 · 置信度 ≥ ${options.confidence} · 匹配 IoU ≥ ${options.match_iou} · 输入 ${options.image_size} px`;
  element("comparisonModelInfo").replaceChildren(...["old", "new"].map(role => {
    const model = models[role];
    const overlap = model.recorded_overlap;
    const source = overlap
      ? `该轮训练/验证重叠 ${overlap.images} 张 · ${overlap.groups} 个同源组${overlap.folders ? ` · ${overlap.folders} 个目录` : ""}`
      : "未能核实训练数据";
    const info = document.createElement("p");
    info.className = "muted";
    info.textContent = `${role === "old" ? "旧版" : "新版"} ${model.filename} · SHA-256 ${model.sha256.slice(0, 12)} · ${source}`;
    return info;
  }));
  const metrics = ["recall", "missed", "false_positives"];
  element("comparisonClassRows").innerHTML = classes.map(({name, old, new: current}) => {
    const values = [["old", old], ["new", current]].flatMap(([role, row]) => metrics.map(key => {
      const value = row[key];
      const change = row === current && value !== null && old[key] !== null
        ? Math.sign(value - old[key]) * (key === "recall" ? 1 : -1) : 0;
      const style = change > 0 ? ' class="result-better"' : change < 0 ? ' class="result-worse"' : "";
      if (report.has_details && key !== "recall") {
        const kind = `${role}_${key}`;
        const description = escapeHtml(`${name} ${comparisonReviewLabels[kind]} ${value} 个，查看图片`);
        return `<td${style}><button type="button" class="comparison-count" data-review-kind="${kind}"
          data-review-label="${escapeHtml(name)}" data-empty="${value === 0}" aria-label="${description}">${value}</button></td>`;
      }
      const text = value === null ? "—" : key === "recall" ? recallFormat.format(value) : value;
      return `<td${style}>${text}</td>`;
    })).join("");
    return `<tr><th scope="row">${escapeHtml(name)}</th><td>${old.ground_truth}</td>${values}</tr>`;
  }).join("");
  comparisonUi.reportData = report;
  element("openComparisonReview").hidden = !report.has_details;
  const hint = element("comparisonReviewHint");
  hint.hidden = Boolean(report.has_details);
  hint.textContent = "早期对比只保存了汇总。用同一盲测集重新对比，即可逐图回看。";
  element("downloadComparison").href = projectUrl(`/comparisons/${run.id}/report`);
  element("comparisonReport").hidden = false;
}

async function showComparisonRun() {
  const run = comparisonUi.runs.find(record => record.id === comparisonUi.selected);
  const active = comparisonActive.has(run?.status);
  const progress = element("comparisonProgress");
  progress.hidden = !active;
  element("stopComparison").hidden = !active || run.status === "cancelling";
  element("comparisonReport").hidden = !run || comparisonUi.report !== run.id;
  if (!run) {
    comparisonMessage("comparisonRunStatus", "启动对比后可在此查看进度与结果。");
    return;
  }
  const done = run.progress?.done ?? 0;
  const total = run.progress?.total ?? 2 * run.images;
  if (active) {
    progress.max = total;
    progress.value = done;
  }
  const message = run.error || (active ? `${comparisonStatuses[run.status]} · ${done} / ${total} 次图片推理` : comparisonStatuses[run.status]);
  comparisonMessage("comparisonRunStatus", message, run.status === "failed");
  if (run.status !== "completed" || [comparisonUi.report, comparisonUi.loadingReport].includes(run.id)) return;
  comparisonUi.loadingReport = run.id;
  const session = comparisonUi.session;
  try {
    const report = await api(projectUrl(`/comparisons/${run.id}/report`));
    if (session !== comparisonUi.session || run.id !== comparisonUi.selected) return;
    renderComparisonReport(run, report);
    comparisonUi.report = run.id;
  } finally {
    if (session === comparisonUi.session && comparisonUi.loadingReport === run.id) comparisonUi.loadingReport = null;
  }
}

async function refreshComparisons() {
  if (!element("comparisonDialog").open) return;
  const read = ++comparisonUi.read;
  const runs = await api(projectUrl("/comparisons"));
  if (read !== comparisonUi.read || !element("comparisonDialog").open) return;
  comparisonUi.runs = runs;
  const select = element("comparisonRunSelect");
  select.hidden = !runs.length;
  const options = runs.map(run => `<option value="${run.id}">${escapeHtml(run.set_name)} · ${new Date(run.created_at).toLocaleString()} · ${comparisonStatuses[run.status]}</option>`).join("");
  if (options !== select.dataset.options) {
    select.innerHTML = options;
    select.dataset.options = options;
  }
  if (!runs.some(run => run.id === comparisonUi.selected)) comparisonUi.selected = runs[0]?.id || "";
  select.value = comparisonUi.selected;
  syncComparisonSet();
  await showComparisonRun();
}

async function openComparisonDialog() {
  if (!state.project) return;
  const session = ++comparisonUi.session;
  if (comparisonUi.project !== state.project.id) {
    comparisonUi.project = state.project.id;
    comparisonUi.sets = [];
    comparisonUi.runs = [];
    comparisonUi.selected = "";
    element("comparisonOldPath").value = state.project.model_path || "";
    element("comparisonNewPath").value = "";
    element("comparisonSet").value = "";
  }
  comparisonUi.report = comparisonUi.reportData = comparisonUi.loadingReport = null;
  element("comparisonModels").replaceChildren();
  element("comparisonRunSelect").hidden = true;
  element("comparisonProgress").hidden = true;
  element("stopComparison").hidden = true;
  element("comparisonReport").hidden = true;
  comparisonMessage("comparisonRunStatus", "正在读取对比记录…");
  comparisonMessage("comparisonMessage", "");
  renderBlindsets();
  element("comparisonDialog").showModal();
  const refresh = () => refreshComparisons().catch(error => {
    if (session === comparisonUi.session) comparisonMessage("comparisonRunStatus", error.message, true);
  });
  comparisonUi.timer = setInterval(refresh, 1500);
  await Promise.all([refresh(), api(projectUrl("/blindsets")).then(sets => {
    if (session !== comparisonUi.session) return;
    comparisonUi.sets = sets;
    renderBlindsets();
  }), api(projectUrl("/comparison-models")).then(models => {
    if (session !== comparisonUi.session) return;
    element("comparisonModels").replaceChildren(...models.map(model => {
      const option = new Option(model.label, model.path);
      option.label = model.label;
      return option;
    }));
  })]).catch(error => {
    if (session === comparisonUi.session) comparisonMessage("comparisonMessage", error.message, true);
  });
}

function clearComparisonFrame() {
  comparisonReview.imageSequence++;
  comparisonReview.frame = null;
  element("comparisonFrame").hidden = true;
  for (const id of ["comparisonFrameOldImage", "comparisonFrameNewImage"]) element(id).replaceChildren();
  if (comparisonReview.imageUrl) URL.revokeObjectURL(comparisonReview.imageUrl);
  comparisonReview.imageUrl = null;
}

function comparisonBox(box, frame, kind, caption, focused=false) {
  const [left, top, right, bottom] = box;
  const node = document.createElement("div");
  node.className = `comparison-frame-box ${kind}${focused ? " focused" : ""}`;
  Object.assign(node.style, {left:`${100 * left / frame.width}%`, top:`${100 * top / frame.height}%`,
    width:`${100 * (right - left) / frame.width}%`, height:`${100 * (bottom - top) / frame.height}%`});
  node.title = caption;
  const text = document.createElement("span");
  text.textContent = caption;
  node.append(text);
  return node;
}

function renderComparisonFrame() {
  const frame = comparisonReview.frame;
  if (!frame) return;
  const name = element("comparisonReviewClass").value;
  const kind = element("comparisonReviewKind").value;
  const allClasses = element("comparisonFrameAllClasses").checked;
  const visible = box => allClasses || box.label === name;
  const truth = frame.truth.map((box, index) => ({...box, index})).filter(visible);
  const hits = {old:new Set(frame.old.map(box => box.truth_index)), new:new Set(frame.new.map(box => box.truth_index))};
  element("comparisonFrameName").textContent = frame.photo_id;
  for (const [role, prefix] of [["old", "comparisonFrameOld"], ["new", "comparisonFrameNew"]]) {
    const predictions = frame[role].filter(visible);
    const misses = truth.filter(box => !hits[role].has(box.index)).length;
    const errors = predictions.filter(box => box.truth_index === null).length;
    element(`${prefix}Counts`).textContent = `${predictions.length} 个预测 · ${misses} 个漏检 · ${errors} 个误检`;
    const scene = document.createElement("div");
    scene.className = "comparison-frame-scene";
    scene.style.setProperty("--image-ratio", frame.width / frame.height);
    const picture = document.createElement("img");
    picture.src = comparisonReview.imageUrl;
    picture.alt = `冻结图片 ${frame.photo_id}`;
    const overlay = document.createElement("div");
    overlay.className = "comparison-frame-overlay";
    overlay.setAttribute("aria-hidden", "true");
    for (const box of truth) {
      const missed = !hits[role].has(box.index);
      const oldHit = hits.old.has(box.index), newHit = hits.new.has(box.index);
      const added = box.label === name && kind === "newly_missed" && oldHit && !newHit;
      const fixed = box.label === name && kind === "fixed_missed" && !oldHit && newHit;
      const status = added && role === "new" ? " · 新增漏检" : fixed && role === "new" ? " · 修复漏检" : missed ? " · 漏检" : "";
      overlay.append(comparisonBox(box.box, frame, "truth", box.label + status, added || fixed));
    }
    for (const prediction of predictions) {
      const style = prediction.truth_index === null ? "false-positive" : "matched";
      const caption = `${prediction.label} ${recallFormat.format(prediction.confidence)}`;
      const box = comparisonBox(prediction.box, frame, style, caption);
      box.title = prediction.iou === null ? `${caption} · 误检` : `${caption} · 匹配 IoU ${prediction.iou.toFixed(2)}`;
      overlay.append(box);
    }
    scene.append(picture, overlay);
    element(`${prefix}Image`).replaceChildren(scene);
  }
  element("comparisonFrame").hidden = false;
}

async function selectComparisonFrame(id) {
  clearComparisonFrame();
  const sequence = comparisonReview.imageSequence;
  const url = `${comparisonReview.url}/${encodeURIComponent(id)}`;
  const isCurrent = () => element("comparisonReviewDialog").open && sequence === comparisonReview.imageSequence;
  for (const button of element("comparisonReviewList").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.reviewFrame === id));
  }
  comparisonMessage("comparisonFrameStatus", "正在读取冻结图片和标注…");
  let imageUrl;
  try {
    const [frame, image] = await Promise.all([api(url), fetch(`${url}/image`).then(async response => {
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "冻结图片无法读取");
      }
      return response.blob();
    })]);
    if (!isCurrent()) return;
    imageUrl = URL.createObjectURL(image);
    const preview = new Image();
    preview.src = imageUrl;
    await preview.decode();
    if (!isCurrent()) return;
    comparisonReview.frame = frame;
    comparisonReview.imageUrl = imageUrl;
    imageUrl = null;
    renderComparisonFrame();
    comparisonMessage("comparisonFrameStatus", "");
  } catch (error) {
    if (isCurrent()) comparisonMessage("comparisonFrameStatus", error.message, true);
  } finally {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  }
}

async function loadComparisonReview(offset=0) {
  clearComparisonFrame();
  const sequence = ++comparisonReview.listSequence;
  comparisonReview.page = null;
  const list = element("comparisonReviewList");
  list.replaceChildren();
  list.scrollTop = 0;
  element("comparisonReviewPages").hidden = true;
  comparisonMessage("comparisonReviewStatus", "正在筛选图片…");
  comparisonMessage("comparisonFrameStatus", "");
  const kind = element("comparisonReviewKind").value;
  const query = new URLSearchParams({label:element("comparisonReviewClass").value, kind, offset});
  const isCurrent = () => element("comparisonReviewDialog").open && sequence === comparisonReview.listSequence;
  try {
    const page = await api(`${comparisonReview.url}?${query}`);
    if (!isCurrent()) return;
    comparisonReview.page = page;
    const metric = kind.endsWith("false_positives") ? "误检" : "漏检";
    const shown = page.total > page.limit ? ` · 第 ${page.offset + 1}–${page.offset + page.items.length} 张` : "";
    comparisonMessage("comparisonReviewStatus", `${page.total} 张图 · ${page.count} 个${metric}${shown}`);
    element("comparisonReviewPages").hidden = page.total <= page.limit;
    element("comparisonReviewPrevious").disabled = page.offset === 0;
    element("comparisonReviewNext").disabled = page.offset + page.limit >= page.total;
    list.innerHTML = page.items.map(item => `<button type="button" class="button" data-review-frame="${escapeHtml(item.id)}"
      aria-pressed="false"><span>${escapeHtml(item.photo_id)}</span><span>${item.count} 个</span></button>`).join("");
    if (page.items.length) {
      await selectComparisonFrame(page.items[0].id);
    } else {
      comparisonMessage("comparisonFrameStatus", "此筛选条件没有对应图片。");
    }
  } catch (error) {
    if (isCurrent()) comparisonMessage("comparisonReviewStatus", error.message, true);
  }
}

async function openComparisonReview(label, kind="new_missed") {
  const report = comparisonUi.reportData;
  if (!report?.has_details || comparisonUi.report !== comparisonUi.selected) return;
  comparisonReview.url = projectUrl(`/comparisons/${comparisonUi.report}/frames`);
  element("comparisonReviewRun").textContent = report.snapshot.name;
  element("comparisonReviewClass").replaceChildren(...report.classes.map(row => new Option(row.name, row.name)));
  if (label !== undefined) element("comparisonReviewClass").value = label;
  element("comparisonReviewKind").replaceChildren(...Object.entries(comparisonReviewLabels).map(([value, text]) => new Option(text, value)));
  element("comparisonReviewKind").value = kind;
  element("comparisonFrameOldModel").textContent = `旧版 · ${report.models.old.filename}`;
  element("comparisonFrameNewModel").textContent = `新版 · ${report.models.new.filename}`;
  element("comparisonFrameAllClasses").checked = false;
  element("comparisonReviewDialog").showModal();
  await loadComparisonReview();
}

async function startComparison(event) {
  event.preventDefault();
  if (element("startComparison").disabled) return;
  const session = comparisonUi.session;
  comparisonUi.submitting = true;
  syncComparisonSet();
  comparisonMessage("comparisonMessage", "正在创建对比任务…");
  try {
    const run = await api(projectUrl("/comparisons"), "POST", {set_id:element("comparisonSet").value,
      old_path:element("comparisonOldPath").value.trim(), new_path:element("comparisonNewPath").value.trim(),
      confidence:element("comparisonConfidence").valueAsNumber, match_iou:element("comparisonIoU").valueAsNumber,
      image_size:Number(element("comparisonImageSize").value)});
    if (session !== comparisonUi.session) return;
    comparisonUi.selected = run.id;
    comparisonMessage("comparisonMessage", "");
    await refreshComparisons();
  } catch (error) {
    if (session === comparisonUi.session) comparisonMessage("comparisonMessage", error.message, true);
  } finally {
    comparisonUi.submitting = false;
    syncComparisonSet();
  }
}

function openBlindsetDialog() {
  element("blindsetBatch").innerHTML = element("exportBatch").innerHTML;
  element("blindsetBatch").value = filters().batch;
  element("blindsetUnit").value = state.project.has_source_groups ? "source_group" : "image";
  comparisonMessage("blindsetMessage", "");
  element("blindsetDialog").showModal();
  element("blindsetName").focus();
}

async function freezeBlindset(event) {
  event.preventDefault();
  if (comparisonUi.freezing) return;
  const session = comparisonUi.session;
  comparisonUi.freezing = true;
  element("freezeBlindset").disabled = true;
  comparisonMessage("blindsetMessage", "正在复制图片并固定标注版本…");
  try {
    const snapshot = await api(projectUrl("/blindsets"), "POST", {name:element("blindsetName").value.trim(),
      batch:element("blindsetBatch").value, unit:element("blindsetUnit").value,
      percent:element("blindsetPercent").valueAsNumber, seed:element("blindsetSeed").valueAsNumber});
    if (session !== comparisonUi.session) return;
    comparisonUi.sets = [snapshot, ...comparisonUi.sets.filter(set => set.id !== snapshot.id)];
    renderBlindsets(snapshot.id);
    element("blindsetDialog").close();
    element("blindsetName").value = "";
    comparisonMessage("comparisonMessage", `已冻结盲测集「${snapshot.name}」。`);
  } catch (error) {
    if (session === comparisonUi.session) {
      comparisonMessage(element("blindsetDialog").open ? "blindsetMessage" : "comparisonMessage", error.message, true);
    }
  } finally {
    comparisonUi.freezing = false;
    element("freezeBlindset").disabled = false;
  }
}

async function exportDataset() {
  const plan = exportPreview.plan;
  if (!element("exportDialog").open || !plan || plan.data.blocking_issue || exportPreview.running) return;
  if (plan.data.unreviewed && !window.confirm(`这次导出包含 ${plan.data.unreviewed} 张待审核图片。确定生成此导出包？`)) return;
  exportPreview.running = true;
  element("exportControls").disabled = true;
  element("refreshExportPreview").disabled = true;
  element("createExport").disabled = true;
  const status = element("exportResult");
  status.dataset.error = "false";
  status.textContent = "正在核对原图并生成 ZIP…";
  let failed = false;
  try {
    const result=await api(plan.url, "POST", {...plan.options, preview_token:plan.data.preview_token});
    const counts=result.split_counts;
    const summary=counts ? `训练 ${counts.train} 张 / 验证 ${counts.val} 张 / 测试 ${counts.test} 张。` : "未划分数据集。";
    const augmented = result.augmented_images ? `原图 ${result.original_images} 张，训练增强 ${result.augmented_images} 张。` : "";
    const verified = result.integrity_check?.passed ? "标注完整性校验通过。" : "";
    status.innerHTML=`<p class="muted">已导出 ${result.images} 张。${summary}${augmented}${verified}<a href="${escapeHtml(result.url)}" download>再次下载 ZIP</a></p>`;
    const link=document.createElement("a");link.href=result.url;link.download=result.filename;document.body.append(link);link.click();link.remove();
    toast(`已生成 ${result.images} 张数据的导出包`);
  } catch (error) {
    failed = true;
    status.dataset.error = "true";
    status.textContent = error.message;
  } finally {
    exportPreview.running = false;
    element("exportControls").disabled = false;
    element("refreshExportPreview").disabled = false;
    if (failed) updateExportPreview();
    else element("createExport").disabled = !exportPreview.plan;
  }
}
async function createProject() {
  if(state.busy || !await ensureSaved())return;
  element("createProject").disabled=true;
  try{
    const project=await api("/api/projects","POST",{name:element("newProjectName").value.trim(),image_root:element("newImageRoot").value.trim(),
      ...(projectImages.files.length ? {image_files:projectImages.files} : {}),
      annotation_root:element("newAnnotationRoot").value.trim(),preannotation_root:element("newPreannotationRoot").value.trim(),
      classes:element("newClasses").value.split(/\r?\n/).map(name=>name.trim()).filter(Boolean),class_map_path:element("newClassMap").value.trim(),
      model_path:element("newModelPath").value.trim(),manifest_path:element("newManifest").value.trim()});
    state.projects.push(project);
    element("projectSelect").add(new Option(project.name,project.id));
    element("projectDialog").close();
    await selectProject(project.id);
    toast("项目已接入，源文件未改动");
  }finally{element("createProject").disabled=false;}
}

listen("projectSelect","change",event=>selectProject(event.target.value));
listen("imageList","click",event=>{const button=event.target.closest("[data-item]");if(button)return loadItem(button.dataset.item);});
listen("objectList","click",event=>{const button=event.target.closest("[data-shape]");if(button && !state.busy)selectShape(Number(button.dataset.shape));});
listen("classList","click",event=>{const button=event.target.closest("[data-class]");if(button){state.activeClass=Number(button.dataset.class);renderClasses();}});
document.querySelectorAll("[data-tool]").forEach(button=>button.addEventListener("click",()=>{if(!state.busy)setTool(button.dataset.tool);}));
for(const id of ["batchSelect","statusSelect"])listen(id,"change",async()=>{
  if(!state.project || state.busy || !await ensureSaved())return;
  localStorage.setItem(`labelpilot-batch-${state.project.id}`,element("batchSelect").value);
  state.offset=0;await refreshList();
  if(state.items[0])await loadItem(state.items[0].id);
  else if(managedQueue())finishReviewQueue();
});
listen("reviewQueueButton", "click", async () => {
  if (!state.project || state.busy || !await ensureSaved()) return;
  const status = element("statusSelect");
  status.value = reviewModes.has(status.value) ? "" : "review";
  state.offset = 0;
  await refreshList();
  if (state.items[0]) await loadItem(state.items[0].id);
  else if (reviewModes.has(status.value)) finishReviewQueue();
});
listen("searchInput","input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.offset=0;refreshList().catch(error=>toast(error.message,true));},220);});
listen("refreshButton","click",async()=>{
  if(state.busy || !await ensureSaved())return;
  const selectedId=state.current?.item.id;
  const summary=await api(projectUrl("/rescan"),"POST",{});
  Object.assign(state.project,summary);
  const options='<option value="">全部批次</option>'+summary.batches.map(batch=>`<option value="${escapeHtml(batch)}">${escapeHtml(batch)}</option>`).join("");
  const previousBatch=element("batchSelect").value;
  element("batchSelect").innerHTML=options;element("exportBatch").innerHTML=options;
  element("batchSelect").value=summary.batches.includes(previousBatch)?previousBatch:"";
  state.offset=0;await refreshList();
  if(selectedId && state.items.some(item=>item.id===selectedId))await loadItem(selectedId);
  else if(state.items[0])await loadItem(state.items[0].id);
  toast("目录和源标注已重新扫描；本工具的已保存修改保留");
});
listen("prevPage","click",async()=>{if(state.busy || !await ensureSaved())return;state.offset=Math.max(0,state.offset-state.pageSize);await refreshList();});
listen("nextPage","click",async()=>{if(state.busy || !await ensureSaved())return;state.offset+=state.pageSize;await refreshList();});
listen("saveButton","click",()=>saveCurrent("draft"));
listen("reviewButton","click",()=>navigateImage(1,true));
listen("previousImage","click",()=>navigateImage(-1));
listen("nextImage","click",()=>navigateImage(1));
listen("skipButton", "click", async () => {
  if (!await saveCurrent("skipped")) return;
  if (managedQueue() && !state.items.length) finishReviewQueue();
  else await navigateImage(1);
});
listen("undoButton","click",()=>undo());listen("redoButton","click",()=>undo(true));
listen("zoomOut","click",()=>zoom(.8));listen("zoomIn","click",()=>zoom(1.25));listen("fitButton","click",fitImage);
listen("focusButton","click",()=>toggleFocusMode());
element("labelForm").addEventListener("submit",event=>{event.preventDefault();addCustomLabel().catch(error=>toast(error.message,true));});
element("pickerLabelForm").addEventListener("submit",event=>{event.preventDefault();applyPickedLabel().catch(error=>toast(error.message,true));});
element("labelPicker").addEventListener("keydown",event=>{
  if(event.key.toLowerCase()!=="q" || event.ctrlKey || event.metaKey || event.altKey || event.isComposing)return;
  if(["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName) || event.target.isContentEditable)return;
  event.preventDefault();
  if(event.repeat || state.busy || state.labelPickerSaving)return;
  const button=event.target.closest("button");
  if(button)button.click();
  else element("pickerLabelForm").requestSubmit(element("confirmPickedLabel"));
});
listen("pickerLabelInput","input",renderLabelOptions);
listen("pickerLabelOptions","click",event=>{const button=event.target.closest("[data-label]");if(button)return applyPickedLabel(button.dataset.label);});
for(const id of ["cancelPickedLabel","closeLabelPicker"])listen(id,"click",cancelLabelPicker);
element("labelPicker").addEventListener("cancel",event=>{event.preventDefault();cancelLabelPicker();});
element("labelPicker").addEventListener("close",()=>{if(element("labelPicker").open)return;state.pendingShape=null;draw();canvas.focus({preventScroll:true});});
window.addEventListener("resize",positionLabelPicker);
listen("deleteShape","click",deleteShape);
listen("shapeClass","change",event=>{if(state.selected<0 || state.busy)return;checkpoint();state.current.document.shapes[state.selected].label=event.target.value;changed();});
listen("shapeDescription","change",event=>{if(state.selected<0 || state.busy)return;checkpoint();state.current.document.shapes[state.selected].description=event.target.value;changed();});
listen("shapeDifficult","change",event=>{if(state.selected<0 || state.busy)return;checkpoint();state.current.document.shapes[state.selected].difficult=event.target.checked;changed();});
listen("toggleLabels","click",()=>{state.showLabels=!state.showLabels;draw();});
listen("objectsTab","click",()=>{element("objectsContent").hidden=false;element("classesContent").hidden=true;element("objectsTab").classList.add("selected");element("classesTab").classList.remove("selected");});
listen("classesTab","click",()=>{element("objectsContent").hidden=true;element("classesContent").hidden=false;element("objectsTab").classList.remove("selected");element("classesTab").classList.add("selected");});
for(const id of ["modelSettings","modelSettingsSmall"])listen(id,"click",openModelDialog);
listen("predictButton","click",predictCurrent);listen("singlePredict","click",predictCurrent);
listen("saveModel","click",saveModelConfig);listen("batchPredict","click",startBatch);
listen("browseModelFile","click",()=>openModelBrowser("file"));
listen("browseModelFolder","click",()=>openModelBrowser("folder"));
listen("modelPath","input",modelSelectionChanged);
element("modelBrowserPathForm").addEventListener("submit",event=>{event.preventDefault();if(!modelBrowser.loading)loadModelDirectory(element("modelBrowserPath").value.trim());});
listen("modelBrowserDrive","change",event=>{if(event.target.value)return loadModelDirectory(event.target.value);});
listen("modelBrowserUp","click",()=>loadModelDirectory(modelBrowser.listing.parent));
listen("searchModelFolder","click",()=>loadModelDirectory(modelBrowser.listing.path,true));
listen("modelBrowserList","click",event=>{
  if(modelBrowser.loading)return;
  const folder=event.target.closest("[data-model-directory]");
  if(folder)return loadModelDirectory(folder.dataset.modelDirectory);
  const file=event.target.closest("[data-model-file]");
  if(file)chooseModelFile(file.dataset.modelFile);
});
listen("cancelModelBrowser","click",()=>element("modelBrowserDialog").close());
element("modelBrowserDialog").addEventListener("close",()=>{if(element("modelBrowserDialog").open)return;modelBrowser.sequence++;modelBrowser.loading=false;});
listen("confidence","input",event=>{element("confidenceValue").value=Number(event.target.value).toFixed(2);});
listen("iou","input",event=>{element("iouValue").value=Number(event.target.value).toFixed(2);});
listen("discardPrediction","click",()=>{state.prediction=null;element("predictionDialog").close();});
listen("appendPrediction","click",()=>applyPrediction(true));listen("replacePrediction","click",()=>applyPrediction(false));
listen("jobsList","click",async event=>{const button=event.target.closest("[data-cancel-job]");if(button){await api(`/api/jobs/${button.dataset.cancelJob}/cancel`,"POST",{});toast("已请求取消；当前推理结束后停止，不会写入该张结果");}});
for (const id of ["railExport", "exportButton"]) listen(id, "click", () => openExportDialog());
listen("cleanButton", "click", () => openExportDialog(true));
for (const id of ["exportFormat", "exportBatch", "reviewedOnly", "includeImages", "cleanDataset", "splitDataset",
  "exportSplitUnit", "splitTrain", "splitVal", "splitTest", "splitSeed", "augmentationPreset", "augmentationCopies", "augmentationSeed"]) listen(id, "input", () => {
  element("exportResult").textContent = "";
  updateExportPreview();
});
listen("augmentationPreset", "change", updateExportPreview);
listen("refreshExportPreview", "click", updateExportPreview);
listen("exportCleaningExamples", "click", async event => {
  const button = event.target.closest("[data-clean-item]");
  if (!button || state.busy || !await ensureSaved()) return;
  element("exportDialog").close();
  await loadItem(button.dataset.cleanItem);
});
element("exportDialog").addEventListener("close", () => {
  if (element("exportDialog").open) return;
  clearTimeout(exportPreview.timer);
  exportPreview.sequence++;
  exportPreview.plan = null;
});
listen("createExport","click",exportDataset);
listen("trainingButton", "click", openTrainingDialog);
listen("trainingForm", "submit", startTraining);
element("trainingDialog").addEventListener("close", () => {
  if (element("trainingDialog").open) return;
  clearInterval(trainingTimer);
  trainingSequence++;
});
listen("trainingRuns", "click", async event => {
  const cancel = event.target.closest("[data-cancel-training]");
  if (cancel) {
    cancel.disabled = true;
    try {
      await api(projectUrl(`/trainings/${cancel.dataset.cancelTraining}/cancel`), "POST", {});
      await refreshTrainingRuns();
    } finally { cancel.disabled = false; }
    return;
  }
  const use = event.target.closest("[data-use-training]");
  if (!use) return;
  const run = trainingRuns?.find(record => record.id === use.dataset.useTraining);
  if (!run?.weights) return;
  use.disabled = true;
  try {
    const project = await api(projectUrl("/model"), "POST", {path:run.weights});
    Object.assign(state.project, project);
    element("modelPath").value = project.model_path;
    element("trainingMode").value = "fine_tune";
    trainingMessage("已将该轮训练的权重设为项目预标注模型。");
  } finally { use.disabled = false; }
});
listen("comparisonButton", "click", openComparisonDialog);
listen("comparisonForm", "submit", startComparison);
listen("comparisonSet", "change", syncComparisonSet);
listen("browseComparisonOld", "click", () => openModelBrowser("file", "comparisonOldPath"));
listen("browseComparisonNew", "click", () => openModelBrowser("file", "comparisonNewPath"));
listen("openBlindset", "click", openBlindsetDialog);
listen("blindsetForm", "submit", freezeBlindset);
listen("comparisonRunSelect", "change", event => {
  comparisonUi.selected = event.target.value;
  return showComparisonRun();
});
listen("comparisonClassRows", "click", event => {
  const count = event.target.closest("[data-review-kind]");
  if (count) return openComparisonReview(count.dataset.reviewLabel, count.dataset.reviewKind);
});
listen("openComparisonReview", "click", () => openComparisonReview());
for (const id of ["comparisonReviewClass", "comparisonReviewKind"]) {
  listen(id, "change", () => loadComparisonReview());
}
listen("comparisonReviewList", "click", event => {
  const selected = event.target.closest("[data-review-frame]");
  if (selected) return selectComparisonFrame(selected.dataset.reviewFrame);
});
listen("comparisonFrameAllClasses", "change", renderComparisonFrame);
listen("comparisonReviewPrevious", "click", () => {
  const page = comparisonReview.page;
  if (page?.offset > 0) return loadComparisonReview(Math.max(0, page.offset - page.limit));
});
listen("comparisonReviewNext", "click", () => {
  const page = comparisonReview.page;
  if (page && page.offset + page.limit < page.total) return loadComparisonReview(page.offset + page.limit);
});
element("comparisonReviewDialog").addEventListener("close", () => {
  if (element("comparisonReviewDialog").open) return;
  comparisonReview.listSequence++;
  clearComparisonFrame();
});
listen("stopComparison", "click", async () => {
  const run = comparisonUi.selected;
  if (!run) return;
  element("stopComparison").disabled = true;
  try {
    await api(projectUrl(`/comparisons/${run}/cancel`), "POST", {});
    await refreshComparisons();
  } finally { element("stopComparison").disabled = false; }
});
element("comparisonDialog").addEventListener("close", () => {
  if (element("comparisonDialog").open) return;
  if (element("comparisonReviewDialog").open) element("comparisonReviewDialog").close();
  clearInterval(comparisonUi.timer);
  comparisonUi.session++;
  comparisonUi.read++;
});
listen("newProjectButton","click",()=>element("projectDialog").showModal());listen("createProject","click",createProject);
listen("chooseImageFiles","click",()=>openImageBrowser("file"));
listen("chooseImageFolder","click",()=>openImageBrowser("folder"));
listen("newImageRoot","input",()=>{projectImages.files=[];updateImageSelection();});
listen("useEntireImageFolder","click",()=>{projectImages.files=[];updateImageSelection();});
element("imageBrowserPathForm").addEventListener("submit",event=>{event.preventDefault();if(!imageBrowser.loading)loadImageDirectory(element("imageBrowserPath").value.trim());});
listen("imageBrowserDrive","change",event=>{if(event.target.value)return loadImageDirectory(event.target.value);});
listen("imageBrowserUp","click",()=>loadImageDirectory(imageBrowser.listing.parent));
listen("imageBrowserList","click",event=>{
  if(imageBrowser.loading || !imageBrowser.valid)return;
  const folder=event.target.closest("[data-image-directory]");
  if(folder)return loadImageDirectory(folder.dataset.imageDirectory);
  const file=event.target.closest("[data-image-file]");
  if(!file || imageBrowser.mode!=="file")return;
  const path=file.dataset.imageFile;
  if(imageBrowser.selected.has(path))imageBrowser.selected.delete(path);
  else {if(imageBrowser.selected.size>=10000)throw new Error("单次最多选择 10000 张图片");imageBrowser.selected.add(path);}
  file.setAttribute("aria-pressed",String(imageBrowser.selected.has(path)));
  updateImageBrowserControls();
});
listen("confirmImageSelection","click",confirmImageSelection);
listen("cancelImageBrowser","click",()=>element("imageBrowserDialog").close());
element("imageBrowserDialog").addEventListener("close",()=>{if(element("imageBrowserDialog").open)return;imageBrowser.sequence++;imageBrowser.loading=false;imageBrowser.valid=false;});
listen("historyButton","click",showHistory);listen("historyList","click",event=>{const button=event.target.closest("[data-restore]");if(button)return restoreRevision(Number(button.dataset.restore));});
for(const id of ["helpButton","keyboardHelp"])listen(id,"click",()=>element("helpDialog").showModal());
listen("workspaceButton","click",()=>{document.querySelectorAll("dialog[open]").forEach(dialog=>dialog.close());});
window.addEventListener("beforeunload",event=>{if(state.dirty){event.preventDefault();event.returnValue="";}});
window.addEventListener("blur",()=>{state.space=false;state.canvasPointer=null;updateCrosshair();});
document.addEventListener("keyup",event=>{if(event.code==="Space"){state.space=false;updateCrosshair();}});
document.addEventListener("keydown",event=>{
  const key=event.key.toLowerCase();
  if(event.isComposing)return;
  if((event.ctrlKey || event.metaKey) && key==="s" && !document.querySelector("dialog[open]")){
    event.preventDefault();
    if(!state.busy){document.activeElement.blur();Promise.resolve().then(()=>saveCurrent("draft"));}
    return;
  }
  if(["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName) || event.target.isContentEditable || document.querySelector("dialog[open]"))return;
  if(key==="tab" && !event.ctrlKey && !event.metaKey && !event.altKey){event.preventDefault();toggleFocusMode();return;}
  if(event.code==="Space"){event.preventDefault();state.space=true;updateCrosshair();return;}
  if(state.busy)return;
  if((event.ctrlKey || event.metaKey) && key==="z"){event.preventDefault();undo(event.shiftKey);return;}
  if(event.ctrlKey || event.metaKey || event.altKey)return;
  if(efficiency.keydown(event,key)){event.preventDefault();return;}
  if(event.repeat && ["a","d","e","q"].includes(key))return;
  if(key==="v")setTool("select");if(key==="r")setTool("rectangle");if(key==="p")setTool("polygon");if(key==="h")setTool("pan");if(key==="f")fitImage();
  if(key==="delete" || key==="backspace"){event.preventDefault();deleteShape();}
  if(key==="escape"){
    if(state.drag?.before)state.current.document=state.drag.before;
    state.drag=null;state.rectangle=null;state.polygon=[];state.selected=-1;renderObjects();draw();
  }
  if((key==="enter" || key==="q") && state.polygon.length){event.preventDefault();finishPolygon();}
  if(key==="l"){state.showLabels=!state.showLabels;draw();}
  if(key==="a")navigateImage(-1).catch(error=>toast(error.message,true));
  if(key==="d")navigateImage(1).catch(error=>toast(error.message,true));
  if(key==="e"){
    event.preventDefault();
    navigateImage(1,true).catch(error=>toast(error.message,true));
  }
  if(key==="?")element("helpDialog").showModal();
  const number=Number(key);
  if(number>=1 && number<=9 && number<=state.project?.classes.length){state.activeClass=number-1;renderClasses();}
});

toggleFocusMode(localStorage.getItem("labelpilot-focus")==="true");
efficiency.install();
efficiency.render();
workspaceResume.install();
automationUi.install();
initialize().catch(error=>{toast(error.message,true);showCanvasMessage("工作区启动失败",error.message);});
