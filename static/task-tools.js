"use strict";

const taskUi = (() => {
  const names = {detect:"目标检测", segment:"实例分割", semantic:"语义分割", hand_pose:"手部姿态", ocr:"OCR 文字与读数"};
  const instructions = {
    ocr:"R 框文字 · P 顺序标四角、Q 完成 · 填写原文与可读性 · E 审核下一张",
    detect:"R 两次点击画框 · Esc 取消 · A / D 翻页",
    segment:"R 画定位框 · SAM3 按框生成轮廓 · P 描轮廓 · Q 完成",
    semantic:"P 圈区域 · B 画笔 · X 橡皮 · 按类别生成像素掩膜",
    hand_pose:"K 添加手部 · 按提示标关键点 · 可跳过不可见点",
  };
  const maskLayer = document.createElement("canvas");
  const maskContext = maskLayer.getContext("2d");
  const decodedMasks = new Map();
  let stroke = null;
  let newHand = null;
  let joint = 0;
  let placing = null;

  function type() { return state.project?.task_type || "detect"; }
  function name() { return names[type()]; }
  function isMaskTask() { return type() === "segment" || type() === "semantic"; }
  function selectedHand() {
    const shape = state.current?.document.shapes[state.selected];
    return shape?.shape_type === "hand" ? shape : null;
  }
  function projectTask() { return document.querySelector('input[name="projectTask"]:checked').value; }
  function projectSubtitle() {
    element("projectSubtitle").textContent = state.project.batches.length + " 个批次 · " + state.project.classes.length + " 个类别";
  }

  function projectChanged() {
    const task = type();
    document.body.dataset.task = task;
    const chip = element("projectTask");
    chip.textContent = name();
    chip.hidden = false;
    projectSubtitle();
    document.querySelectorAll("[data-tasks]").forEach(node => {
      const available = node.dataset.tasks.split(" ").includes(task);
      node.hidden = !available;
      if (node.tagName === "OPTION") node.disabled = !available;
    });
    element("canvasHint").textContent = instructions[task];
    element("classesHint").textContent = instructions[task] + " · 右侧 × 删除未使用标签";
    element("modelTaskHint").textContent = {
      detect:"选择类别顺序一致的目标检测权重。",
      segment:"可用检测权重先定位，再由 SAM3 按框生成轮廓；也可直接使用实例分割权重。矩形框可存草稿，补全轮廓后才能审核。",
      semantic:"使用语义分割权重。背景 ID 为 0，前景类别按项目顺序从 1 开始。",
      hand_pose:"使用 21 关键点的手部姿态权重。模板顺序从手腕到小指。",
      ocr:"使用本地 PaddleOCR 3.2.0（PP-OCRv5 检测 + 识别）生成文字区域和原文建议；每个结果都需核对。",
    }[task];
    element("predictButton").textContent = task === "ocr" ? "✦ OCR 预标注" : "✦ 模型预标注";
    element("singlePredict").textContent = task === "ocr" ? "✦ OCR 预标注当前图片" : "✦ 预标注当前图片";
    element("batchPredict").textContent = task === "ocr" ? "批量 OCR 预标注" : "批量预标注";
    const formats = element("exportFormat");
    element("exportOutputTask").value = task === "segment" ? "segment" : "detect";
    formats.querySelector('[value="yolo"]').textContent = {detect:"YOLO · 按标注用途导出", segment:"YOLO · 按标注用途导出", hand_pose:"YOLO Pose · 21 关键点"}[task] || "YOLO";
    formats.querySelector('[value="coco"]').textContent = task === "hand_pose" ? "COCO Keypoints" : "COCO";
    if (!formats.selectedOptions.length || formats.selectedOptions[0].disabled) {
      formats.value = Array.from(formats.options).find(option => !option.disabled).value;
    }
    if (task === "ocr") formats.value = "paddleocr";
    element("augmentationSettings").hidden = !["detect", "segment"].includes(task);
    if (element("augmentationSettings").hidden) element("augmentationPreset").value = "none";
    element("acceptPrompt").textContent = isMaskTask() ? "采用分割建议" : "采用候选框";
    element("jointSelect").replaceChildren(...(state.project.keypoints || []).map((point, index) =>
      new Option(String(index + 1) + " · " + point.label, index)));
    itemChanged();
  }

  function itemChanged() {
    stroke = newHand = placing = null;
    joint = 0;
    state.sam3Candidates = [];
    state.sam3PreviewIndex = -1;
    document.dispatchEvent(new Event("labelpilot-item-changed"));
    decodedMasks.clear();
    render();
  }

  function toolAllowed(tool) {
    const node = document.querySelector('[data-tool="' + tool + '"]');
    return !node || !node.hidden;
  }

  function toolChanged() {
    stroke = newHand = placing = null;
    render();
  }

  function selectionChanged() {
    placing = null;
    joint = 0;
  }

  function render() {
    ocrUi.render();
    const hand = selectedHand();
    const handTool = type() === "hand_pose" && state.tool === "hand";
    const next = newHand?.points.length || 0;
    const ready = Boolean(state.current) && !state.busy;
    element("brushControls").hidden = type() !== "semantic" || !["brush", "erase"].includes(state.tool);
    element("handTools").hidden = type() !== "hand_pose";
    if (type() === "hand_pose") {
      const hint = handTool ? (next + 1) + " / 21 · 点击" + state.project.keypoints[next].label : "K 添加一只手，或选中骨架调整关键点";
      element("handStep").textContent = placing ? "点击画布定位 " + state.project.keypoints[joint].label : hint;
    }
    element("skipHandPoint").hidden = element("undoHandPoint").hidden = element("finishHand").hidden = !handTool;
    element("skipHandPoint").disabled = !ready;
    element("undoHandPoint").disabled = !ready || !next;
    element("finishHand").disabled = !ready || !newHand?.points.some(point => point[2]);
    element("handPointEditor").hidden = !hand;
    if (hand) {
      element("jointSelect").value = joint;
      element("jointVisibility").value = placing || hand.keypoints[joint][2];
      element("placeJoint").textContent = placing ? "取消定位" : "在画布定位";
      element("placeJoint").disabled = element("jointSelect").disabled = element("jointVisibility").disabled = !ready;
    }
  }

  function handBounds(points) {
    const present = points.filter(point => point[2]);
    const xs = present.map(point => point[0]);
    const ys = present.map(point => point[1]);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const pad = Math.max(3, .08 * Math.max(right - left, bottom - top));
    return [[Math.max(0, left - pad), Math.max(0, top - pad)],
      [Math.min(state.current.width, right + pad), Math.min(state.current.height, bottom + pad)]];
  }

  function completeHand() {
    if (!newHand || state.busy) return;
    const {points, label} = newHand;
    if (!points.some(point => point[2])) {
      newHand = null;
      render();
      draw();
      return toast("请至少标注一个关键点");
    }
    while (points.length < state.project.keypoints.length) points.push([0, 0, 0]);
    newHand = null;
    joint = 0;
    addShape(handBounds(points), "hand", {label, keypoints:points});
  }

  function nextJoint(point) {
    if (state.tool !== "hand" || !state.current || state.busy) return;
    newHand ||= {points:[], label:state.project.classes[state.activeClass].name};
    newHand.points.push(point);
    if (newHand.points.length === state.project.keypoints.length) completeHand();
    else { render(); draw(); }
  }

  function previousJoint() {
    if (!newHand) return;
    newHand.points.pop();
    if (!newHand.points.length) newHand = null;
    render();
    draw();
  }

  function setJoint(hand, index, point) {
    hand.keypoints[index] = point;
    hand.points = handBounds(hand.keypoints);
  }

  function pointerDown(event, point) {
    if (state.tool === "brush" || state.tool === "erase") {
      stroke = {
        shape_type:"brush", label:state.project.classes[state.activeClass].name, points:[point],
        erase:state.tool === "erase", radius:Number(element("brushRadius").value),
        pointer:event.pointerId,
      };
      canvas.setPointerCapture(event.pointerId);
      draw();
      return true;
    }
    if (state.tool === "hand") {
      nextJoint([...point, 2]);
      return true;
    }
    if (state.tool !== "select" || type() !== "hand_pose") return false;
    const hand = selectedHand();
    if (!hand) return false;
    if (placing) {
      checkpoint();
      setJoint(hand, joint, [...point, placing]);
      placing = null;
      changed();
      return true;
    }
    const hit = hand.keypoints.findIndex(([x, y, visible]) => visible && Math.hypot(x - point[0], y - point[1]) * state.scale < 9);
    if (hit < 0) return false;
    joint = hit;
    canvas.setPointerCapture(event.pointerId);
    state.drag = {type:"hand-point", before:snapshot()};
    render();
    draw();
    return true;
  }

  function pointerMove(event, point) {
    if (stroke && stroke.pointer === event.pointerId) {
      const prior = stroke.points.at(-1);
      if (Math.hypot(prior[0] - point[0], prior[1] - point[1]) < 1) return true;
      if (stroke.points.length === 4096) stroke.points = stroke.points.filter((_, index) => index % 2 === 0);
      stroke.points.push(point);
      draw();
      return true;
    }
    if (state.drag?.type !== "hand-point") return false;
    const hand = selectedHand();
    setJoint(hand, joint, [...point, hand.keypoints[joint][2]]);
    draw();
    return true;
  }

  function pointerUp(event) {
    if (!stroke || stroke.pointer !== event.pointerId) return false;
    const {points, shape_type:kind, pointer, ...style} = stroke;
    stroke = null;
    addShape(points, kind, style);
    return true;
  }

  function cancelPointer() { stroke = null; }
  function blocksReview() { return Boolean(stroke || newHand || placing || state.drag?.type === "hand-point"); }

  function keydown(key) {
    const tool = {b:"brush", x:"erase", k:"hand"}[key];
    if (tool && toolAllowed(tool)) { setTool(tool); return true; }
    if (key === "backspace" && newHand) { previousJoint(); return true; }
    if (key === "enter" && newHand) { completeHand(); return true; }
    if (key === "escape" && (stroke || newHand || placing)) {
      stroke = newHand = placing = null;
      render();
      draw();
      return true;
    }
    return false;
  }

  function opacity() { return Number(element("maskOpacity").value) / 100; }

  function coloredMask(source, color) {
    let stored = decodedMasks.get(source);
    if (!stored) {
      stored = {mask:null, tint:new Map()};
      decodedMasks.set(source, stored);
      const image = new Image();
      image.onload = () => {
        if (decodedMasks.get(source) !== stored) return;
        const surface = document.createElement("canvas");
        surface.width = image.width;
        surface.height = image.height;
        const painter = surface.getContext("2d", {willReadFrequently:true});
        painter.drawImage(image, 0, 0);
        stored.mask = painter.getImageData(0, 0, image.width, image.height);
        draw();
      };
      image.src = "data:image/png;base64," + source;
    }
    if (!stored.mask) return null;
    if (stored.tint.has(color)) return stored.tint.get(color);
    const surface = document.createElement("canvas");
    surface.width = stored.mask.width;
    surface.height = stored.mask.height;
    const painter = surface.getContext("2d");
    const tint = painter.createImageData(surface.width, surface.height);
    const rgb = [1, 3, 5].map(start => parseInt(color.slice(start, start + 2), 16));
    for (let i = 0; i < tint.data.length; i += 4) {
      tint.data[i] = rgb[0];
      tint.data[i + 1] = rgb[1];
      tint.data[i + 2] = rgb[2];
      tint.data[i + 3] = stored.mask.data[i];
    }
    painter.putImageData(tint, 0, 0);
    stored.tint.set(color, surface);
    return surface;
  }

  function drawMask(ctx, shape, color) {
    const source = coloredMask(shape.mask, color);
    if (source) ctx.drawImage(source, ...shape.points[0]);
  }

  function paintRegion(ctx, shape, color) {
    ctx.fillStyle = ctx.strokeStyle = color;
    if (shape.shape_type === "mask") { drawMask(ctx, shape, color); return; }
    ctx.beginPath();
    ctx.moveTo(...shape.points[0]);
    for (const point of shape.points.slice(1)) ctx.lineTo(...point);
    if (shape.shape_type === "polygon") {
      ctx.closePath();
      ctx.fill();
      return;
    }
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = 2 * shape.radius + 1;
    if (shape.points.length > 1) ctx.stroke();
    else {
      ctx.beginPath();
      ctx.arc(...shape.points[0], shape.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawRegions() {
    if (type() !== "semantic") return;
    const ratio = devicePixelRatio;
    if (maskLayer.width !== canvas.width || maskLayer.height !== canvas.height) {
      maskLayer.width = canvas.width;
      maskLayer.height = canvas.height;
    }
    maskContext.setTransform(1, 0, 0, 1, 0, 0);
    maskContext.clearRect(0, 0, maskLayer.width, maskLayer.height);
    maskContext.save();
    const scale = ratio * state.scale;
    maskContext.setTransform(scale, 0, 0, scale, ratio * state.panX, ratio * state.panY);
    maskContext.beginPath();
    maskContext.rect(0, 0, state.current.width, state.current.height);
    maskContext.clip();
    maskContext.imageSmoothingEnabled = false;
    for (const shape of state.current.document.shapes.concat(stroke ? [stroke] : [])) {
      maskContext.globalCompositeOperation = shape.erase ? "destination-out" : "source-over";
      paintRegion(maskContext, shape, classInfo(shape.label).color);
    }
    maskContext.restore();
    context.save();
    context.globalAlpha = opacity();
    context.drawImage(maskLayer, 0, 0, canvas.width / ratio, canvas.height / ratio);
    context.restore();
  }

  function drawMaskPreview(shape, color) {
    context.save();
    context.translate(state.panX, state.panY);
    context.scale(state.scale, state.scale);
    context.globalAlpha = .55;
    context.imageSmoothingEnabled = false;
    drawMask(context, shape, color);
    context.restore();
  }

  function drawHand(keypoints, color, selected=false) {
    context.save();
    context.strokeStyle = context.fillStyle = color;
    context.lineWidth = selected ? 2.5 : 1.7;
    for (const [start, end] of state.project.skeleton) {
      const a = keypoints[start], b = keypoints[end];
      if (!a?.[2] || !b?.[2]) continue;
      context.setLineDash(a[2] === 1 || b[2] === 1 ? [3, 3] : []);
      pathPoints([a, b], false);
      context.stroke();
    }
    context.setLineDash([]);
    keypoints.forEach((point, index) => {
      if (!point[2]) return;
      context.beginPath();
      context.arc(...screenPoint(point), selected && index === joint ? 5 : 3.3, 0, Math.PI * 2);
      if (point[2] === 2) context.fill();
      else context.stroke();
      if (selected && index === joint) {
        context.strokeStyle = "#fff";
        context.stroke();
        context.strokeStyle = color;
      }
    });
    context.restore();
  }

  function drawDraft() {
    if (newHand) drawHand(newHand.points, classInfo(newHand.label).color);
  }

  function install() {
    ocrUi.install();
    document.querySelectorAll('[name="projectTask"]').forEach(node => node.addEventListener("change", () => {
      element("newClasses").placeholder = projectTask() === "hand_pose" ? "hand（留空也会使用此类别）"
        : projectTask() === "ocr" ? "text（留空也会使用此类别）" : "class_a\nclass_b";
    }));
    listen("maskOpacity", "input", draw);
    listen("brushRadius", "input", () => { element("brushRadiusValue").textContent = element("brushRadius").value + " px"; });
    listen("skipHandPoint", "click", () => nextJoint([0, 0, 0]));
    listen("undoHandPoint", "click", previousJoint);
    listen("finishHand", "click", completeHand);
    listen("jointSelect", "change", event => {
      joint = Number(event.target.value);
      placing = null;
      render();
      draw();
    });
    listen("placeJoint", "click", () => {
      const hand = selectedHand();
      if (!hand || state.busy) return;
      const value = placing ? null : hand.keypoints[joint][2] || 2;
      if (state.tool !== "select") setTool("select");
      placing = value;
      render();
    });
    listen("jointVisibility", "change", event => {
      const hand = selectedHand();
      if (!hand || state.busy) return;
      const next = Number(event.target.value);
      const [x, y, current] = hand.keypoints[joint];
      if (next && !current) {
        if (state.tool !== "select") setTool("select");
        placing = next;
        return render();
      }
      if (!next && !hand.keypoints.some((point, index) => index !== joint && point[2])) {
        render();
        return toast("每只手至少保留一个已标关键点");
      }
      checkpoint();
      setJoint(hand, joint, next ? [x, y, next] : [0, 0, 0]);
      placing = null;
      changed();
    });
  }

  return {type, name, isMaskTask, projectTask, projectSubtitle, projectChanged, itemChanged, toolAllowed,
    toolChanged, selectionChanged, render, pointerDown, pointerMove, pointerUp, cancelPointer, blocksReview, keydown,
    drawRegions, drawMaskPreview, drawHand, drawDraft, opacity, instructions, install};
})();
