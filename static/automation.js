"use strict";

const automationUi = (() => {
  const controls = {
    intake:{folder:"workflowFolder", watch:"workflowWatch", similarity:"workflowSimilarity", representatives:"workflowRepresentatives",
      companion:"workflowCompanion", confidence:"workflowConfidence", low:"workflowLow", rare:"workflowRare", sample:"workflowSample"},
    release:{blindset:"workflowBlindset", batch:"workflowTrainBatch", enabled:"workflowAutoTrain", min_images:"workflowMinImages",
      min_each:"workflowMinEach", baseline:"workflowBaseline", mode:"workflowTrainMode", epochs:"workflowEpochs",
      unit:"workflowSplitUnit", eval_confidence:"workflowEvalConfidence", eval_iou:"workflowEvalIou"}
  };
  const stages = {scan:"等待读取新图", curate:"相似图分组", predict:"批量预标注", route:"审核分流",
    prepare:"核对训练数据", train:"本地训练", compare:"固定盲测", done:"完成"};
  const statuses = {queued:"已排队", waiting:"等待", running:"运行中", completed:"已完成", partial:"部分待重试",
    failed:"待重试", passed:"验收通过", rejected:"未达验收门槛"};
  const names = {intake:"新批次处理", routing:"智能审核分流", release:"训练与验收"};
  const reasonNames = {empty:"无候选框", disagree:"两模型分歧", rare:"稀有类别", low:"低置信度"};
  let tab = "intake", recipe, runs = [], timer, session = 0, pending = false, lastMarkup = "";

  function routeText(route) {
    const labels = route.reasons.map(name => reasonNames[name]);
    if (!labels.length) labels.push("普通候选");
    if (route.sample) labels.push("来源抽样");
    return labels.join(" · ");
  }

  function message(text, error=false) { comparisonMessage("workflowStatus", text, error); }

  function setTab(type) {
    tab = type;
    for (const [kind, label] of [["intake","Intake"],["release","Release"]]) {
      const active = kind === type;
      element(`automation${label}Tab`).setAttribute("aria-pressed", String(active));
      element(`automation${label}Panel`).hidden = !active;
    }
    renderButtons();
  }

  function renderButtons() {
    element("saveWorkflow").disabled = !recipe || pending;
    for (const [kind, id] of [["intake","runIntake"],["routing","runRouting"],["release","runRelease"]]) {
      element(id).disabled = !recipe || pending || runs.some(run => run.kind === kind && ["queued","waiting","running"].includes(run.status));
    }
  }

  function populate() {
    for (const [group, fields] of Object.entries(controls)) {
      for (const [key, id] of Object.entries(fields)) {
        const node = element(id);
        if (node.type === "checkbox") node.checked = Boolean(recipe[group][key]);
        else node.value = key === "folder" && recipe[group][key] === "." ? "" : recipe[group][key];
      }
    }
    element("workflowGates").innerHTML = state.project.classes.map(({name}) => {
      const rule = recipe.release.gates[name];
      return `<tr data-gate="${escapeHtml(name)}"><th scope="row"><label class="check"><input type="checkbox" data-include-gate ${rule ? "checked" : ""}>${escapeHtml(name)}</label></th>
        <td><input aria-label="${escapeHtml(name)} 最低召回率" data-gate-recall type="number" min="0" max="100" step="any" value="${rule ? (rule.recall * 100).toFixed(2) : "80"}"></td>
        <td><input aria-label="${escapeHtml(name)} 误检上限" data-gate-fp type="number" min="0" max="1000000" step="1" value="${rule?.false_positives ?? 5}"></td></tr>`;
    }).join("");
  }

  function currentRecipe(includeBlindset) {
    const result = copy(recipe);
    for (const [group, fields] of Object.entries(controls)) {
      for (const [key, id] of Object.entries(fields)) {
        const node = element(id);
        if (node.type === "number" && (!node.checkValidity() || !Number.isFinite(node.valueAsNumber))) {
          setTab(group);
          node.reportValidity();
          throw new Error("请填写有效的工作流数值");
        }
        result[group][key] = node.type === "checkbox" ? node.checked : node.type === "number" ? node.valueAsNumber : node.value.trim();
      }
    }
    if ((includeBlindset || result.release.enabled) && !result.release.blindset) {
      setTab("release");
      throw new Error("请先选择固定盲测集");
    }
    result.release.gates = Object.fromEntries([...element("workflowGates").querySelectorAll("[data-gate]")].flatMap(row => {
      if (!row.querySelector("[data-include-gate]").checked) return [];
      const recall = row.querySelector("[data-gate-recall]"), fp = row.querySelector("[data-gate-fp]");
      if (![recall,fp].every(node => node.checkValidity() && Number.isFinite(node.valueAsNumber))) {
        setTab("release");
        throw new Error("请检查已选类别的验收门槛");
      }
      return [[row.dataset.gate, {recall:recall.valueAsNumber / 100, false_positives:fp.valueAsNumber}]];
    }));
    return result;
  }

  function listRecords(data) {
    runs = data.runs;
    const readiness = data.readiness;
    const title = readiness ? `可训练 ${readiness.images} / ${readiness.required} 张${Object.keys(readiness.missing).length
      ? " · 覆盖不足 " + Object.entries(readiness.missing).map(([name,count]) => `${name} ${count} 张`).join("、") : ""}` : "";
    comparisonMessage("workflowReadiness", data.watch_error || title, Boolean(data.watch_error));
    const markup = runs.map(run => {
      const job = run.progress;
      const counts = [];
      if (run.included !== undefined) counts.push(`纳入 ${run.included} 张 · 精确重复 ${run.duplicates} 张`);
      if (run.similar !== undefined) counts.push(`相似组 ${run.similar} · 暂缓 ${run.deferred} 张`);
      if (run.routed !== undefined) counts.push(`待审核 ${run.routed} 张 · 抽样 ${run.samples} 张`);
      if (run.reviewed !== undefined) counts.push(`训练候选 ${run.reviewed} 张`);
      if (job?.total !== undefined) counts.push(`已处理 ${(job.done || 0) + (job.skipped || 0) + (job.failed || 0)} / ${job.total} 张`);
      if (job?.epochs) counts.push(`训练 ${job.epoch || 0} / ${job.epochs} 轮`);
      if (run.unreadable?.length) counts.push(`暂不可读 ${run.unreadable.length} 张`);
      const actions = [];
      if (run.routed) actions.push(["triage","智能审核"], ["spot_review","来源抽检"]);
      if (run.included) actions.push(["curated","筛后队列"]);
      if (run.comparison_id && run.gates) actions.push(["report","查看盲测结果"]);
      if (["failed","partial"].includes(run.status)) actions.push(["retry","重试待处理项"]);
      if (run.status === "passed") actions.push(["activate","确认启用候选模型"]);
      const notes = run.error || run.message || (run.unreadable?.length ? run.unreadable[0].relative + " · " + run.unreadable[0].error : "");
      return `<article class="training-run" data-workflow-run="${run.id}">
        <div class="training-run-heading"><strong>${names[run.kind]} · ${statuses[run.status]}</strong><time>${new Date(run.created_at).toLocaleString()}</time></div>
        <p class="muted">${escapeHtml(run.batch || "全部批次")} · ${stages[run.phase]}</p>
        ${counts.length ? `<p class="muted">${counts.join(" · ")}</p>` : ""}
        ${notes ? `<p class="workflow-run-note">${escapeHtml(notes)}</p>` : ""}
        ${run.gates?.length ? `<div class="workflow-gate-results">${run.gates.map(row => `<p data-passed="${row.passed}">
          ${row.passed ? "通过" : "未通过"} · ${escapeHtml(row.name)} · 召回 ${row.recall === null ? "无真值" : recallFormat.format(row.recall)}
          / ${recallFormat.format(row.rule.recall)} · 误检 ${row.false_positives ?? "—"} / ${row.rule.false_positives}</p>`).join("")}</div>` : ""}
        ${run.activated_at ? `<p class="muted">已启用候选模型 · ${new Date(run.activated_at).toLocaleString()}</p>` : ""}
        ${actions.length ? `<div class="training-run-actions">${actions.map(([action,label]) => `<button type="button" class="button" data-workflow-action="${action}">${label}</button>`).join("")}</div>` : ""}
      </article>`;
    }).join("") || '<p class="muted">暂无工作流记录</p>';
    if (markup !== lastMarkup) {
      element("workflowRuns").innerHTML = markup;
      lastMarkup = markup;
    }
    renderButtons();
  }

  async function refresh(openSession=session) {
    const data = await api(projectUrl("/automation"));
    if (openSession === session && element("automationDialog").open) listRecords(data);
  }

  async function open() {
    if (!state.project || state.busy) return;
    const opened = ++session;
    recipe = null; runs = []; lastMarkup = "";
    element("workflowRuns").textContent = "";
    setTab("intake"); message("正在读取项目配置…");
    element("automationDialog").showModal();
    try {
      const [data, sets] = await Promise.all([api(projectUrl("/automation")),api(projectUrl("/blindsets"))]);
      if (opened !== session || !element("automationDialog").open) return;
      recipe = data.recipe;
      element("workflowTrainBatch").innerHTML = element("exportBatch").innerHTML;
      element("workflowBlindset").replaceChildren(new Option("请选择固定盲测集", ""), ...sets.map(set => new Option(set.name, set.id)));
      populate();
      listRecords(data);
      message("");
      timer = setInterval(() => refresh(opened).catch(error => message(error.message, true)), 1600);
    } catch (error) { if (opened === session) message(error.message, true); }
  }

  async function submit(kind) {
    if (pending || !recipe || !await ensureSaved()) return;
    pending = true;
    renderButtons();
    try {
      const edited = currentRecipe(kind === "release");
      if (kind) {
        const queued = await api(projectUrl("/automation/runs"), "POST", {kind, batch:filters().batch, recipe:edited});
        recipe = queued.recipe;
      } else recipe = await api(projectUrl("/automation"), "PUT", edited);
      message(kind ? "" : "已保存项目配置。");
      await refresh();
    } catch (error) {
      message(error.message, true);
    } finally { pending = false; renderButtons(); }
  }

  async function enterBatch(run, status) {
    if (!await ensureSaved()) return;
    const data = await api("/api/projects");
    const project = data.projects.find(item => item.id === state.project.id);
    if (project) Object.assign(state.project, project);
    const html = '<option value="">全部批次</option>' + state.project.batches.map(batch =>
      `<option value="${escapeHtml(batch)}">${escapeHtml(batch)}</option>`).join("");
    element("batchSelect").innerHTML = element("exportBatch").innerHTML = html;
    element("automationDialog").close();
    await efficiency.enterQueue(status, run.batch);
  }

  async function runAction(event) {
    const button = event.target.closest("[data-workflow-action]");
    const run = runs.find(row => row.id === button?.closest("[data-workflow-run]")?.dataset.workflowRun);
    if (!run || button.disabled) return;
    const action = button.dataset.workflowAction;
    button.disabled = true;
    try {
      if (action === "retry" || action === "activate") {
        const result = await api(projectUrl(`/automation/runs/${run.id}/${action}`), "POST", {});
        if (action === "activate") {
          Object.assign(state.project, result);
          element("modelPath").value = result.model_path;
        }
        message(action === "activate" ? "候选模型已设为项目预标注模型。" : "待处理步骤已重新入队。");
        await refresh();
      } else if (action === "report") {
        element("automationDialog").close();
        await openComparisonDialog();
        comparisonUi.selected = run.comparison_id;
        await refreshComparisons();
      } else await enterBatch(run, action);
    } catch (error) {
      message(error.message, true);
    } finally { button.disabled = false; }
  }

  function install() {
    listen("automationButton", "click", open);
    listen("automationIntakeTab", "click", () => setTab("intake"));
    listen("automationReleaseTab", "click", () => setTab("release"));
    listen("browseWorkflowFolder", "click", () => openImageBrowser("folder", "workflowFolder"));
    listen("browseWorkflowCompanion", "click", () => openModelBrowser("file", "workflowCompanion"));
    listen("browseWorkflowBaseline", "click", () => openModelBrowser("file", "workflowBaseline"));
    listen("saveWorkflow", "click", () => submit());
    listen("runIntake", "click", () => submit("intake"));
    listen("runRouting", "click", () => submit("routing"));
    listen("runRelease", "click", () => submit("release"));
    listen("workflowRuns", "click", runAction);
    element("automationDialog").addEventListener("close", () => {
      if (element("automationDialog").open) return;
      session++;
      clearInterval(timer);
    });
  }

  return {install, routeText};
})();
