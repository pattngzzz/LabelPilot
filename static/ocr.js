"use strict";

const ocrUi = (() => {
  const fields = {Text:"text_raw", Readability:"readability", Role:"value_role", Value:"value_decimal",
    Unit:"unit", Field:"field_name", Channel:"channel", Parent:"parent_id"};
  const defaults = {text_raw:"", readability:"readable", value_role:"unknown", value_decimal:null,
    unit:null, field_name:null, channel:null, parent_id:null};
  const active = () => taskUi.type() === "ocr";

  function form(prefix) {
    return '<canvas id="' + prefix + 'Preview" class="ocr-preview" aria-label="文字区域放大预览"></canvas>' +
      '<label>文字原文（原样保留）<input id="' + prefix + 'Text" maxlength="4096" autocomplete="off" placeholder="如 00.250 g；不要猜测模糊字符"></label>' +
      '<label>可读性<select id="' + prefix + 'Readability"><option value="readable">完整可读</option><option value="partial">部分可读（不用于识别训练）</option><option value="unreadable">不可读（原文留空）</option></select></label>' +
      '<label>字段角色<select id="' + prefix + 'Role"><option value="unknown">未知 / 未确认</option><option value="process_value">实际值 / PV</option><option value="setpoint">设定值 / SV</option><option value="state">状态，如 OFF / Err</option><option value="clock">时钟</option><option value="not_applicable">不适用</option></select></label>' +
      '<details><summary>数值、单位与关联（选填，不自动猜测）</summary>' +
      '<label>十进制数值（保留精度）<input id="' + prefix + 'Value" placeholder="如 00.250，不填单位"></label>' +
      '<label>单位<input id="' + prefix + 'Unit" maxlength="160" placeholder="仅有画面或设备依据时填写"></label>' +
      '<label>字段名称<input id="' + prefix + 'Field" maxlength="160"></label>' +
      '<label>通道<input id="' + prefix + 'Channel" maxlength="160"></label>' +
      '<label>所属面板 / 设备 ID<input id="' + prefix + 'Parent" maxlength="160" placeholder="关联不确定就留空"></label></details>' +
      '<small class="muted">一行文字或一个数值单元一个框。保留小数点、前后零、负号和单位；黑屏不是 0。</small>';
  }

  function populate(prefix, shape) {
    for (const [suffix, field] of Object.entries(fields)) {
      const node = element(prefix + suffix);
      if (document.activeElement !== node) node.value = shape[field] ?? defaults[field] ?? "";
      node.disabled = state.busy;
    }
    preview(prefix, shape.points);
  }

  function values(prefix) {
    return Object.fromEntries(Object.entries(fields).map(([suffix, field]) => {
      const value = element(prefix + suffix).value;
      return [field, field === "text_raw" ? value : value || null];
    }));
  }

  function preview(prefix, points) {
    const surface = element(prefix + "Preview");
    if (!state.image || !points?.length) return;
    const horizontal = points.map(point => point[0]), vertical = points.map(point => point[1]);
    const left = Math.min(...horizontal), top = Math.min(...vertical);
    const width = Math.max(1, Math.max(...horizontal) - left), height = Math.max(1, Math.max(...vertical) - top);
    const scale = Math.min(1, 1800 / width, 500 / height);
    surface.width = Math.max(1, Math.round(width * scale));
    surface.height = Math.max(1, Math.round(height * scale));
    surface.getContext("2d").drawImage(state.image, left, top, width, height, 0, 0, surface.width, surface.height);
  }

  function render() {
    const shape = state.current?.document.shapes[state.selected];
    element("ocrShapeFields").hidden = !active() || !shape;
    if (active() && shape) populate("shapeOcr", shape);
    element("ocrPickerFields").hidden = !active();
  }

  function beginPicker() {
    element("labelPickerTitle").textContent = active() ? "文字与读数标注" : "选择标签";
    element("labelPickerHelp").textContent = active()
      ? "先逐字填写原文及可读性，再按 Enter 或点击保存文字标注；Q 在文本输入框中仍是普通字符。类别名称保持原样，不能把读数当类别。"
      : "Q / Enter 确认当前标签；也可点击已有标签，或点击输入框输入新名称后按 Enter 确认。输入时 q 仍是标签文字。";
    element("ocrPickerFields").hidden = !active();
    if (!active()) return;
    for (const [suffix, field] of Object.entries(fields)) element("pickerOcr" + suffix).value = defaults[field] ?? "";
    preview("pickerOcr", state.pendingShape.points);
  }

  function pickedValues() {
    if (!active()) return {};
    const details = values("pickerOcr");
    if (details.readability === "readable" && !details.text_raw.trim()) throw new Error("请填写准确原文，或设置为部分可读 / 不可读");
    if (details.readability === "unreadable" && details.text_raw) throw new Error("不可读区域请留空原文，不要猜测");
    if (details.readability !== "readable" && details.value_decimal !== null) throw new Error("未完整可读区域请留空确定数值");
    if (/[\r\n\t]/.test(details.text_raw)) throw new Error("每个文字框只标一行，不能输入换行或制表符");
    if (details.value_decimal !== null && !/^-?[0-9]+(?:\.[0-9]+)?$/.test(details.value_decimal)) throw new Error("数值请填写十进制字符串，不含单位");
    return details;
  }

  function install() {
    element("ocrShapeFields").innerHTML = form("shapeOcr");
    element("ocrPickerFields").innerHTML = form("pickerOcr");
    element("ocrPickerFields").querySelector("details").addEventListener("toggle", positionLabelPicker);
    const hint = document.createElement("p");
    hint.className = "task-hint";
    hint.dataset.tasks = "ocr";
    hint.hidden = true;
    hint.textContent = "OCR：PaddleOCR 导出包含检测标签、完整可读文字的识别裁剪及原始属性。部分 / 不可读区域作忽略标记；空白图不生成识别样本。裁剪继承原图划分，建议按来源组划分。当前不提供 OCR 预识别或训练。";
    element("exportFormat").closest(".dialog-body").prepend(hint);
    for (const suffix of Object.keys(fields)) {
      listen("shapeOcr" + suffix, "change", () => {
        if (!active() || state.busy || state.selected < 0) return;
        checkpoint();
        Object.assign(state.current.document.shapes[state.selected], values("shapeOcr"));
        changed();
      });
    }
    document.querySelectorAll('[name="projectTask"]').forEach(node => node.addEventListener("change", () => {
      const ocr = taskUi.projectTask() === "ocr";
      element("newOcrHint").hidden = !ocr;
      if (ocr) element("newClasses").placeholder = "text（留空也会使用此类别）";
    }));
  }

  return {active, render, beginPicker, pickedValues, install};
})();
