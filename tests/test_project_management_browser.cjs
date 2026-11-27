const assert = require("node:assert/strict");
const {spawn, spawnSync} = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {test} = require("node:test");

const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function eventually(callback) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    const result = await callback();
    if (result) return result;
    await sleep(70);
  }
  throw new Error("Project management browser check timed out");
}

test("project archive, restore, delete and same-image OCR reuse", {timeout:120000}, async context => {
  const root = path.resolve(__dirname, "..");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "labelpilot-project-management-"));
  const home = path.join(fixture, "tool");
  const profile = path.join(fixture, "chrome");
  const python = process.env.LABELPILOT_PYTHON || "python";
  const setup = spawnSync(python, ["-c", `
import sys
from pathlib import Path
from PIL import Image
from labelpilot.store import Store
root = Path(sys.argv[1])
images = root / 'images'
images.mkdir()
Image.new('RGB', (320, 200), '#ddeeff').save(images / 'panel.png')
store = Store(root / 'tool')
source = store.register({'name': 'OCR source', 'image_root': str(images), 'classes': ['text'], 'task_type': 'ocr'})
project = source['id']
item = store.project(project).items[0]['id']
current = store.detail(project, item)
current['document']['shapes'] = [{'label': 'text', 'shape_type': 'rectangle', 'points': [[15, 20], [170, 85]], 'text_raw': '25.6', 'unit': 'C', 'readability': 'readable', 'value_role': 'process_value'}]
store.save(project, item, {**current, 'expected_version': 0, 'status': 'reviewed'})
store.management.archive(project)
print(project)
`, fixture], {cwd:root, windowsHide:true, encoding:"utf8"});
  assert.equal(setup.status, 0, setup.stderr || setup.error?.message);
  const sourceId = setup.stdout.trim().split(/\r?\n/).at(-1);
  const listener = net.createServer();
  await new Promise(resolve => listener.listen(0, "127.0.0.1", resolve));
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const url = `http://127.0.0.1:${port}`;
  const server = spawn(python, ["-c", "import sys,uvicorn; from labelpilot.server import create_app; uvicorn.run(create_app(sys.argv[1]),host='127.0.0.1',port=int(sys.argv[2]),log_level='warning')", home, String(port)],
    {cwd:root, windowsHide:true, env:{...process.env, YOLO_CONFIG_DIR:path.join(fixture,"ultralytics"), MPLCONFIGDIR:path.join(fixture,"matplotlib")}, stdio:"ignore"});
  let browser, socket, cdp;
  const errors = [];
  try {
    await eventually(async () => {try {return (await fetch(`${url}/api/health`)).ok;} catch {return false;}});
    browser = spawn(process.env.LABELPILOT_CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe",
      ["--headless=new", "--disable-gpu", "--disable-background-networking", "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--window-size=1600,1000", "about:blank"],
      {windowsHide:true, stdio:"ignore"});
    await eventually(() => fs.existsSync(path.join(profile, "DevToolsActivePort")));
    const debugPort = fs.readFileSync(path.join(profile, "DevToolsActivePort"), "utf8").split("\n")[0];
    const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
    socket = new WebSocket(targets.find(target => target.type === "page").webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let sequence = 0;
    const pending = new Map();
    cdp = (method, params={}) => new Promise((resolve, reject) => {
      const id = ++sequence;
      const timer = setTimeout(() => {pending.delete(id); reject(new Error(`CDP timeout: ${method}`));}, 15000);
      pending.set(id, {resolve, reject, timer});
      socket.send(JSON.stringify({id, method, params}));
    });
    socket.onmessage = message => {
      const data = JSON.parse(message.data);
      const request = pending.get(data.id);
      if (request) {
        clearTimeout(request.timer);
        pending.delete(data.id);
        data.error ? request.reject(new Error(JSON.stringify(data.error))) : request.resolve(data.result);
      }
      if (data.method === "Runtime.exceptionThrown") errors.push(data.params);
      if (data.method === "Page.javascriptDialogOpening") cdp("Page.handleJavaScriptDialog", {accept:true}).catch(() => {});
    };
    const evaluate = async expression => {
      const result = await cdp("Runtime.evaluate", {expression, awaitPromise:true, returnByValue:true});
      assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = expression => eventually(() => evaluate(expression));
    const click = id => evaluate(`element(${JSON.stringify(id)}).click()`);
    await cdp("Page.enable");
    await cdp("Runtime.enable");
    await cdp("Page.navigate", {url});
    await waitFor('typeof projectManagement!=="undefined" && element("projectManagementDialog")?.open && element("managedProjectList").textContent.includes("OCR source")');
    assert.equal(await evaluate('state.project===null && !element("projectDialog").open'), true);
    fs.writeFileSync(path.join(fixture, "archived-projects.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data, "base64"));
    context.diagnostic("All-archived startup exposes recoverable project management");
    await evaluate('element("managedProjectList").querySelector("[data-project-action=restore]").click()');
    await waitFor('state.current?.status==="reviewed" && !state.busy');
    assert.equal(await evaluate('state.current.document.shapes[0].text_raw'), "25.6");
    await evaluate('element("projectManagementDialog").close()');
    await click("projectManagementButton");
    await waitFor('element("projectManagementDialog").open && element("managedProjectList").querySelector("[data-project-action=archive]")!==null');
    await evaluate('element("managedProjectList").querySelector("[data-project-action=archive]").click()');
    await waitFor('state.project===null && !state.busy && element("managedProjectList").querySelector("[data-project-action=delete]")!==null');
    await evaluate('element("managedProjectList").querySelector("[data-project-action=delete]").click()');
    await waitFor('element("deleteProjectDialog").open');
    await evaluate('element("deleteProjectConfirmation").value="OCR source";element("deleteProjectConfirmation").dispatchEvent(new Event("input"))');
    await waitFor('!element("confirmDeleteProject").disabled');
    await click("confirmDeleteProject");
    await waitFor('!element("deleteProjectDialog").open && !state.busy && element("managedProjectCount").textContent.startsWith("0 ")');
    assert.equal(fs.existsSync(path.join(fixture, "images", "panel.png")), true);
    context.diagnostic("Archive and deletion preserve source image; deleted project disappears");
    await evaluate('element("projectManagementDialog").close()');
    await click("newProjectButton");
    await waitFor('element("projectDialog").open');
    await evaluate(`element("newProjectName").value="OCR new project";element("newImageRoot").value=${JSON.stringify(path.join(fixture,"images"))};document.querySelector('input[name="projectTask"][value="ocr"]').checked=true;element("newClasses").value="text";`);
    await click("createProject");
    await waitFor('state.current?.status==="draft" && !state.busy && state.current.document.labelpilot?.annotation_reuse');
    assert.equal(await evaluate('state.current.document.labelpilot.annotation_reuse.project'), sourceId);
    assert.equal(await evaluate('state.current.document.shapes[0].text_raw'), "25.6");
    assert.equal(await evaluate('state.current.document.labelpilot.human_reviewed'), false);
    const previewStatus = await evaluate('(async()=>{const response=await fetch(projectUrl("/export/preview"),{method:"POST",headers:{"Content-Type":"application/json","X-LabelPilot-Token":sessionToken},body:JSON.stringify({format:"paddleocr"})});return response.status})()');
    assert.equal(previewStatus, 422, "Unreviewed reused draft is not eligible for default export");
    await evaluate('element("canvas").focus();document.dispatchEvent(new KeyboardEvent("keydown",{key:"e",bubbles:true}))');
    await waitFor('state.current?.status==="reviewed" && !state.busy');
    assert.equal(await evaluate('state.current.document.labelpilot.human_reviewed'), true);
    assert.equal(await evaluate('element("sourceNote").textContent.includes("本项目已审核")'), true);
    context.diagnostic("OCR reuse survives source-project deletion, remains draft until E review");
    await click("newProjectButton");
    await waitFor('element("projectDialog").open');
    await evaluate('element("newProjectName").value="OCR conflict project"');
    await click("createProject");
    await waitFor('state.current?.status==="unstarted" && !state.busy && !element("reuseAnnotationButton").hidden');
    assert.equal(await evaluate('state.current.document.shapes.length'), 0);
    await click("reuseAnnotationButton");
    await waitFor('element("reuseAnnotationDialog").open && element("reuseAnnotationList").querySelectorAll("[data-reuse-reference]").length===2');
    fs.writeFileSync(path.join(fixture, "reuse-choices.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data, "base64"));
    await evaluate('element("reuseAnnotationList").querySelector("[data-reuse-reference]").click()');
    await waitFor('state.current?.status==="draft" && !state.busy && !element("reuseAnnotationDialog").open');
    assert.equal(await evaluate('state.current.document.shapes[0].text_raw'), "25.6");
    context.diagnostic("Multiple matching sources require explicit selection; no silent overwrite");
    assert.deepEqual(errors, [], "No browser runtime errors");
    fs.writeFileSync(path.join(fixture, "project-management.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data, "base64"));
  } finally {
    if (socket?.readyState === WebSocket.OPEN && cdp) await cdp("Browser.close").catch(() => {});
    socket?.close();
    browser?.kill();
    server.kill();
    context.diagnostic(`Isolated fixtures: ${fixture}`);
  }
});
