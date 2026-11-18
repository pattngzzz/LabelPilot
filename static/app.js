"use strict";

const element = id => document.getElementById(id);
const copy = value => structuredClone(value);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
const statusNames = {unstarted:"未开始",imported:"人工标注 · 待审核",predicted:"模型待审核",draft:"编辑中",reviewed:"已审核",skipped:"已跳过",error:"读取异常"};
const state = {projects:[],project:null,items:[],filtered:0,offset:0,pageSize:60,current:null,image:null,selected:-1,
  activeClass:0,tool:"select",dirty:false,busy:false,undo:[],redo:[],scale:1,panX:0,panY:0,drag:null,
  polygon:[],rectangle:null,pendingShape:null,labelPickerSaving:false,canvasPointer:null,space:false,showLabels:true,prediction:null,loadSequence:0,listSequence:0};
const canvas = element("canvas");
const context = canvas.getContext("2d");
const sessionToken = document.querySelector('meta[name="labelpilot-token"]').content;
const modelBrowser = {mode:"file",listing:null,loading:false,sequence:0};
const imageBrowser = {mode:"folder",listing:null,selected:new Set(),loading:false,valid:false,sequence:0};
const projectImages = {files:[]};
const exportAugmentation = {presets:[],maxCopies:3,loaded:false};
let searchTimer;
let jobTimer;

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
function setBusy(value, message="") {
  state.busy = value;
  for (const id of ["predictButton","singlePredict","batchPredict","saveModel","saveButton","reviewButton","skipButton","previousImage","nextImage","projectSelect","deleteShape","shapeClass","shapeDescription","shapeDifficult","newLabelInput","addLabelButton"]) element(id).disabled = value;
  element("connectionStatus").textContent = value ? message || "正在处理…" : "本地工作区 · 仅本机可访问";
  updateCrosshair();
}

function setDirty(value=true) {
  state.dirty = value;
  element("dirtyBadge").hidden = !value;
  if (value && state.current) setStatus("draft");
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
    element("modelRuntime").textContent = `运行设备：${system.device} · PyTorch ${system.torch}`;
    element("modelCaption").textContent = `${system.cuda ? "CUDA" : "CPU"} · 本地模型 · 按需加载`;
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
  state.activeClass = Math.max(0, state.project.classes.findIndex(item => item.name === "sample_container"));
  localStorage.setItem("labelpilot-project", projectId);
  element("projectSelect").value = projectId;
  element("projectSubtitle").textContent = `${state.project.batches.length} 个批次 · ${state.project.classes.length} 个标签 · 原始数据只读`;
  const options = '<option value="">全部批次</option>' + state.project.batches.map(batch => `<option value="${escapeHtml(batch)}">${escapeHtml(batch)}</option>`).join("");
  element("batchSelect").innerHTML = options;
  element("exportBatch").innerHTML = options;
  const rememberedBatch = localStorage.getItem(`labelpilot-batch-${projectId}`);
  element("batchSelect").value = state.project.batches.includes(rememberedBatch) ? rememberedBatch : state.project.batches.includes("B001") ? "B001" : "";
  element("searchInput").value = "";
  element("statusSelect").value = "";
  element("modelPath").value = state.project.model_path;
  renderClasses();
  await refreshList();
  if (state.items.length) await loadItem(state.items[0].id);
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
}

function renderQueue() {
  element("imageList").innerHTML = state.items.map(item => `<button class="image-row ${state.current?.item.id === item.id ? "selected" : ""}" data-item="${item.id}" title="${escapeHtml(item.relative)}"><img loading="lazy" src="${projectUrl(`/items/${item.id}/image?thumbnail=true`)}" alt=""><span class="image-row-info"><strong>${escapeHtml(item.photo_id)}</strong><small>${escapeHtml(statusNames[item.status])} · ${item.count} 对象</small></span><i class="row-dot ${item.status}"></i></button>`).join("") || '<p class="empty-list">当前筛选暂无图片<br>试试其他批次或状态</p>';
}

async function ensureSaved() {
  if (!state.dirty) return true;
  if (!window.confirm("当前图片有未保存修改。确定保存草稿后继续？取消将留在当前图片。")) return false;
  return saveCurrent("draft");
}

async function loadItem(itemId) {
  if (state.busy || !await ensureSaved()) return;
  const sequence = ++state.loadSequence;
  setBusy(true,"读取原图和标注…");
  showCanvasMessage("正在载入原图…");
  try {
    const current = await api(projectUrl(`/items/${itemId}`));
    const image = new Image();
    image.src = projectUrl(`/items/${itemId}/image`);
    await image.decode();
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
  } catch (error) {
    showCanvasMessage("图片载入失败",error.message);
    throw error;
  } finally { setBusy(false); }
}

async function saveCurrent(status="draft") {
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
    setStatus(response.status);
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
  if (reviewed && (state.rectangle || state.polygon.length || state.pendingShape)) return toast("请先完成或取消当前绘制", true);
  const previousIds = state.items.map(item => item.id);
  const currentIndex = previousIds.indexOf(state.current.item.id);
  const nextId = previousIds[currentIndex + direction];
  if (reviewed && !await saveCurrent("reviewed")) return;
  if (!reviewed && state.dirty && !await saveCurrent("draft")) return;
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
    setTool("select");
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
  if (!state.drag) fitImage();
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
async function openModelBrowser(mode) {
  if(!state.project || state.busy)return;
  modelBrowser.mode=mode;
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
  element("modelPath").value=path;
  modelSelectionChanged();
  element("modelBrowserDialog").close();
  element("modelPath").focus({preventScroll:true});
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
async function openImageBrowser(mode) {
  if(state.busy)return;
  imageBrowser.mode=mode;imageBrowser.listing=null;imageBrowser.selected.clear();imageBrowser.valid=false;
  element("imageBrowserTitle").textContent=mode==="folder" ? "选择待标注图片文件夹" : "选择待标注图片（可多选）";
  element("imageBrowserHelp").textContent=mode==="folder" ? "进入目标目录后点击“选择此文件夹”，导入其中及子目录内的图片。仅引用路径，不上传或修改源文件。" : "点击图片勾选或取消，可选择同一文件夹内的多张图片；切换文件夹会清空本次勾选。支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF。";
  element("imageBrowserList").innerHTML="";
  element("imageBrowserDrive").innerHTML='<option value="">磁盘位置</option>';
  element("imageBrowserPath").value="";
  element("imageBrowserDialog").showModal();
  await loadImageDirectory(element("newImageRoot").value.trim() || state.project?.image_root || null);
}
function confirmImageSelection() {
  if(imageBrowser.loading || !imageBrowser.valid || (imageBrowser.mode==="file" && !imageBrowser.selected.size))return;
  const listing=imageBrowser.listing;
  element("newImageRoot").value=listing.path;
  projectImages.files=imageBrowser.mode==="file" ? listing.files.filter(item=>imageBrowser.selected.has(item.path)).map(item=>item.name) : [];
  updateImageSelection();
  element("imageBrowserDialog").close();
  element("newImageRoot").focus({preventScroll:true});
}
function openModelDialog() {
  if(!state.project)return;
  element("modelPath").value=state.project.model_path;
  modelSelectionChanged();
  element("batchScope").textContent=`范围：${filters().batch || "全部批次"}${filters().query ? " · 搜索 "+filters().query : ""}中的未开始图片（忽略状态筛选）；已有标注不会覆盖`;
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
  if(!window.confirm(`即将对 ${filters().batch || "全部批次"}内最多 ${limit} 张未开始图片运行本地模型。结果仍为待审核，不覆盖任何已有标注。继续？`))return;
  await api(projectUrl("/jobs"),"POST",{...modelOptions(),batch:filters().batch,query:filters().query,limit});
  toast("批量任务已启动，可以继续标注其他图片");
  await pollJobs();
}
async function pollJobs() {
  if(!state.project)return;
  const jobs=await api(projectUrl("/jobs"));
  const jobNames={queued:"等待中",running:"推理中",completed:"已完成",completed_with_errors:"完成，含失败",cancelled:"已取消",interrupted:"启动前任务已中断",failed:"失败"};
  const running=jobs.find(job=>["running","queued"].includes(job.status));
  element("jobStatus").hidden=!running;
  if(running)element("jobStatus").textContent=`✦ 批量预标注 ${running.done+running.skipped+running.failed} / ${running.total} · 待人工审核`;
  element("jobsList").innerHTML=jobs.slice(0,4).map(job=>`<div class="job-row"><div><strong>${jobNames[job.status] || escapeHtml(job.status)} · ${job.done} / ${job.total}</strong><small>跳过 ${job.skipped} · 失败 ${job.failed}${job.errors?.length ? " · "+escapeHtml(job.errors.at(-1).error) : ""}</small></div>${["running","queued"].includes(job.status) ? `<button class="button" data-cancel-job="${job.id}">取消</button>`:""}</div>`).join("");
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
  return {...ratios,seed};
}
function exportAugmentationOptions() {
  const preset=element("augmentationPreset").value;
  if(preset==="none")return null;
  if(!exportAugmentation.presets.some(item=>item.id===preset))throw new Error("请选择有效的数据增强方案");
  if(!element("splitDataset").checked)throw new Error("增强需先划分数据集；仅训练可设为 100 / 0 / 0");
  if(!element("includeImages").checked)throw new Error("数据增强必须勾选包含图片");
  if(!element("reviewedOnly").checked)throw new Error("数据增强仅支持已人工确认的标注");
  const copies=element("augmentationCopies").valueAsNumber;
  if(!Number.isInteger(copies) || copies<1 || copies>exportAugmentation.maxCopies)throw new Error(`额外生成数量必须为 1 到 ${exportAugmentation.maxCopies} 的整数`);
  const seed=element("augmentationSeed").valueAsNumber;
  if(!Number.isInteger(seed) || seed<0 || seed>4294967295)throw new Error("增强随机种子必须为 0 到 4294967295 的整数");
  return {preset,copies,seed};
}
async function openExportDialog() {
  if(!state.project)return;
  updateExportSplit();
  element("exportDialog").showModal();
  if(exportAugmentation.loaded)return;
  try{
    const catalog=await api("/api/augmentation-presets");
    exportAugmentation.presets=catalog.presets;
    exportAugmentation.maxCopies=catalog.max_copies;
    exportAugmentation.loaded=true;
    element("augmentationPreset").innerHTML='<option value="none">不增强 · 保留原样（默认）</option>'+catalog.presets.map(item=>`<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("");
    element("augmentationCopies").max=catalog.max_copies;
    element("augmentationPreset").disabled=false;
    updateExportSplit();
  }catch(error){element("augmentationDescription").textContent=`增强方案加载失败：${error.message}。请重启工具并刷新，仍可不增强导出。`;}
}
function updateExportSplit() {
  element("exportSplitSettings").disabled=!element("splitDataset").checked;
  let invalid=false;
  let message="不划分，按原目录结构导出。";
  try{
    const split=exportSplitOptions();
    if(split)message=`训练 ${split.train}% / 验证 ${split.val}% / 测试 ${split.test}% · 合计 100%`;
  }catch(error){invalid=true;message=error.message;}
  element("exportSplitSummary").textContent=message;
  element("exportSplitSummary").dataset.invalid=String(invalid);
  const enabled=element("augmentationPreset").value!=="none";
  for(const id of ["augmentationCopies","augmentationSeed"])element(id).disabled=!enabled;
  const preset=exportAugmentation.presets.find(item=>item.id===element("augmentationPreset").value);
  element("augmentationDescription").textContent=preset?.description || "不生成额外图片。";
  const presetCount=preset?.preset_count || 1;
  element("augmentationCopiesLabel").textContent=presetCount>1 ? "每张训练图片、每种方案额外生成" : "每张训练图片额外生成";
  let augmentationInvalid=false;
  let augmentationMessage="原图和已有标注只读，默认不增强。";
  try{
    const augmentation=exportAugmentationOptions();
    if(augmentation){
      const extra=augmentation.copies*presetCount;
      const perPreset=presetCount>1 ? `${presetCount} 种方案分别生成，每方案 ${augmentation.copies} 张；` : "";
      augmentationMessage=`保留全部原图，${perPreset}每张训练图片额外生成 ${extra} 张，含原图共 ${extra+1} 张，训练集最终为原来的 ${extra+1} 倍；验证 / 测试张数不变。增强文件是派生标注，未单独人工审核。`;
    }
  }catch(error){augmentationInvalid=true;augmentationMessage=error.message;}
  element("augmentationSummary").textContent=augmentationMessage;
  element("augmentationSummary").dataset.invalid=String(augmentationInvalid);
  invalid=invalid || augmentationInvalid;
  element("createExport").disabled=invalid || element("createExport").dataset.exporting==="true";
}
async function exportDataset() {
  if(!state.project)return;
  if(element("createExport").dataset.exporting==="true")return;
  let split,augmentation;
  try{split=exportSplitOptions();augmentation=exportAugmentationOptions();}catch(error){updateExportSplit();toast(error.message);return;}
  if(!element("reviewedOnly").checked && !window.confirm("你正在导出未审核标注（可能包含模型错误）。确认仅作为待审数据使用，不直接视为真值？"))return;
  element("createExport").dataset.exporting="true";
  element("createExport").disabled=true;
  element("exportResult").textContent=augmentation ? "正在校验图片并生成训练集增强副本和 ZIP，请勿关闭服务…" : "正在校验图片指纹并生成 ZIP，请稍候…";
  try{
    const result=await api(projectUrl("/export"),"POST",{format:element("exportFormat").value,batch:element("exportBatch").value,
      reviewed_only:element("reviewedOnly").checked,include_images:element("includeImages").checked,split,augmentation});
    const counts=result.split_counts;
    const summary=counts ? `训练 ${counts.train} 张 / 验证 ${counts.val} 张 / 测试 ${counts.test} 张。` : "未划分数据集。";
    const augmentationSummary=result.augmented_images ? `其中原图 ${result.original_images} 张，训练增强 ${result.augmented_images} 张（未单独人工审核）。`+(result.integrity_check?.passed ? `标注完整性校验通过：${result.integrity_check.images_checked} 张图片 / ${result.integrity_check.annotations_checked} 个对象。` : "") : "";
    element("exportResult").innerHTML=`<p class="muted">已导出 ${result.images} 张。${summary}${augmentationSummary}<a href="${escapeHtml(result.url)}" download>再次下载 ZIP</a></p>`;
    const link=document.createElement("a");link.href=result.url;link.download=result.filename;document.body.append(link);link.click();link.remove();
    toast(`已生成 ${result.images} 张数据的导出包`);
  }catch(error){element("exportResult").textContent=error.message;throw error;}
  finally{element("createExport").dataset.exporting="false";updateExportSplit();}
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
  if(state.busy || !await ensureSaved())return;
  localStorage.setItem(`labelpilot-batch-${state.project.id}`,element("batchSelect").value);
  state.offset=0;await refreshList();if(state.items[0])await loadItem(state.items[0].id);
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
listen("skipButton","click",async()=>{if(await saveCurrent("skipped"))await navigateImage(1);});
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
for(const id of ["railExport","exportButton"])listen(id,"click",openExportDialog);
listen("splitDataset","change",updateExportSplit);
for(const id of ["splitTrain","splitVal","splitTest","splitSeed"])listen(id,"input",updateExportSplit);
for(const id of ["augmentationPreset","includeImages","reviewedOnly"])listen(id,"change",updateExportSplit);
for(const id of ["augmentationCopies","augmentationSeed"])listen(id,"input",updateExportSplit);
listen("createExport","click",exportDataset);
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
initialize().catch(error=>{toast(error.message,true);showCanvasMessage("工作区启动失败",error.message);});
