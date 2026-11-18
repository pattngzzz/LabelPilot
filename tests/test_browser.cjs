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
    await sleep(50);
  }
  throw new Error("Browser test timed out waiting for application state");
}

test("annotation workflow and local model file/folder selection", {timeout:90000}, async context => {
  const root = path.resolve(__dirname, "..");
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "labelpilot-browser-test-"));
  const images = path.join(home, "images");
  const profile = path.join(home, "chrome");
  const python = process.env.LABELPILOT_PYTHON || "python";
  const chrome = process.env.LABELPILOT_CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
  fs.mkdirSync(images);
  fs.mkdirSync(profile);
  const modelFolder = path.join(home, "模型目录");
  const weightFolder = path.join(modelFolder, "run", "weights");
  fs.mkdirSync(weightFolder, {recursive:true});
  const firstModel = path.join(modelFolder, "best.pt");
  const secondModel = path.join(weightFolder, "final.pt");
  fs.writeFileSync(firstModel, "test fixture; browsing must not deserialize weights");
  fs.writeFileSync(secondModel, "nested test fixture; do not execute");
  fs.writeFileSync(path.join(modelFolder, "ignore.json"), "not a model");
  const fixture = spawnSync(python, ["-c", "from PIL import Image; import sys; from pathlib import Path; [Image.new('RGB',(640,480),'#555577').save(Path(sys.argv[1])/f'image_{index}.png') for index in range(4)]", images], {windowsHide:true, encoding:"utf8"});
  assert.equal(fixture.status, 0, fixture.stderr || fixture.error?.message);
  fs.writeFileSync(path.join(home, "projects.json"), JSON.stringify([{id:"test", name:"Browser test", image_root:images, classes:["class_a", "class_b"], model_path:firstModel}]));
  const listener = net.createServer();
  await new Promise(resolve => listener.listen(0, "127.0.0.1", resolve));
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const url = `http://127.0.0.1:${port}`;
  const server = spawn(python, ["-m", "uvicorn", "labelpilot.server:create_app", "--factory", "--host", "127.0.0.1", "--port", String(port), "--log-level", "warning"], {
    cwd:root, windowsHide:true, env:{...process.env, LABELPILOT_HOME:home, YOLO_CONFIG_DIR:path.join(home,"ultralytics"), MPLCONFIGDIR:path.join(home,"matplotlib")}, stdio:"ignore"
  });
  let browser;
  let socket;
  let cdp;
  const errors = [];
  try {
    await eventually(async () => {try {return (await fetch(`${url}/api/health`)).ok;} catch {return false;}});
    browser = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--window-size=1600,1000", url], {windowsHide:true, stdio:"ignore"});
    const activePort = path.join(profile, "DevToolsActivePort");
    await eventually(() => fs.existsSync(activePort));
    const debugPort = fs.readFileSync(activePort, "utf8").split("\n")[0];
    const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
    socket = new WebSocket(targets.find(target => target.type === "page").webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {socket.onopen=resolve; socket.onerror=reject;});
    let sequence = 0;
    const pending = new Map();
    cdp = (method, params={}) => new Promise((resolve, reject) => {
      const id = ++sequence;
      const timeout = setTimeout(() => {pending.delete(id); reject(new Error(`CDP timeout: ${method}`));}, 7000);
      pending.set(id, {resolve, reject, timeout});
      socket.send(JSON.stringify({id, method, params}));
    });
    socket.onmessage = message => {
      const data = JSON.parse(message.data);
      const request = pending.get(data.id);
      if (request) {
        clearTimeout(request.timeout);
        pending.delete(data.id);
        data.error ? request.reject(new Error(JSON.stringify(data.error))) : request.resolve(data.result);
      }
      if (data.method === "Runtime.exceptionThrown") errors.push(data.params);
      if (data.method === "Page.javascriptDialogOpening") cdp("Page.handleJavaScriptDialog", {accept:false});
    };
    const evaluate = async expression => {
      const output = await cdp("Runtime.evaluate", {expression, awaitPromise:true, returnByValue:true});
      assert.equal(output.exceptionDetails, undefined, JSON.stringify(output.exceptionDetails));
      return output.result.value;
    };
    const check = async (expression, message) => {assert.equal(await evaluate(expression), true, message); context.diagnostic(message);};
    const waitFor = async expression => {
      try {return await eventually(() => evaluate(expression));}
      catch (error) {throw new Error(`${error.message}: ${expression}\n${JSON.stringify(await evaluate('({picker:element("labelPicker").open,error:element("pickerLabelError").textContent,input:element("pickerLabelInput").value,busy:state.busy,saving:state.labelPickerSaving})'))}`);}
    };
    const press = async (key, code) => {
      const windowsVirtualKeyCode = {Escape:27, Enter:13, Tab:9}[key] || key.toUpperCase().charCodeAt(0);
      await cdp("Input.dispatchKeyEvent", {type:"keyDown", key, code, windowsVirtualKeyCode, ...(key === "Enter" ? {text:"\r"} : {})});
      await cdp("Input.dispatchKeyEvent", {type:"keyUp", key, code, windowsVirtualKeyCode});
    };
    const mouseClick = async position => {
      await cdp("Input.dispatchMouseEvent", {type:"mousePressed", button:"left", clickCount:1, ...position});
      await cdp("Input.dispatchMouseEvent", {type:"mouseReleased", button:"left", clickCount:1, ...position});
    };
    const click = async selector => mouseClick(await evaluate(`(()=>{const rect=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};})()`));
    const point = coordinates => evaluate(`(()=>{const position=screenPoint(${JSON.stringify(coordinates)});const rect=canvas.getBoundingClientRect();return {x:rect.left+position[0],y:rect.top+position[1]};})()`);
    const box = async (start=[100,100], end=[350,300]) => {
      await press("r", "KeyR");
      await mouseClick(await point(start));
      const endPosition = await point(end);
      await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...endPosition});
      await mouseClick(endPosition);
      await waitFor('element("labelPicker").open');
    };
    await cdp("Page.enable");
    await cdp("Runtime.enable");
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await waitFor('typeof state!=="undefined" && state.current && !state.busy');
    await press("r", "KeyR");
    await mouseClick(await point([100,100]));
    await check('!!state.rectangle && !state.dirty && state.current.document.shapes.length===0', "First click remains active after releasing the mouse");
    await mouseClick(await point([100,100]));
    await check('!!state.rectangle && !element("labelPicker").open', "Repeated clicks at the same point do not create an empty rectangle");
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...await point([350,300])});
    await check('Math.abs(state.rectangle.end[0]-350)<1', "Rectangle follows the pointer without holding a button");
    await press("e", "KeyE");
    await check('!!state.rectangle && state.current.version===0', "E cannot review an unfinished rectangle");
    await mouseClick(await point([350,300]));
    await waitFor('element("labelPicker").open');
    await check('state.current.document.shapes.length===0 && !state.dirty', "Pending rectangle is not saved before a label is chosen");
    await click('[data-label="class_a"]');
    await waitFor('!element("labelPicker").open');
    await check('state.current.document.shapes.length===1 && state.current.document.shapes[0].label==="class_a"', "Existing original label applies on click");
    await click("#undoButton");
    await check('state.current.document.shapes.length===0', "New rectangle is one undo operation");
    await click("#redoButton");
    await box([350,300], [100,100]);
    await press("Escape", "Escape");
    await check('!state.pendingShape && !element("labelPicker").open && state.current.document.shapes.length===1', "Escape cancels the pending rectangle without changing existing objects");
    await box([350,300], [100,100]);
    await cdp("Input.insertText", {text:"rade_custom"});
    await press("e", "KeyE");
    await check('element("labelPicker").open && state.current.version===0', "E does not review while entering a label");
    await evaluate('window.testApi=api;api=async(path,method,...args)=>{if(method==="POST")throw new Error("simulated label failure");return testApi(path,method,...args);};');
    await click("#confirmPickedLabel");
    await check('!!state.pendingShape && element("labelPicker").open && element("pickerLabelError").textContent.includes("simulated label failure") && state.current.document.shapes.length===1', "Failed label creation preserves the pending rectangle for retry");
    await evaluate('api=testApi;');
    await press("Enter", "Enter");
    await waitFor('!element("labelPicker").open');
    await check('state.project.classes.length===3 && state.project.classes[2].name==="rade_custom" && state.current.document.shapes.at(-1).label==="rade_custom"', "Enter creates an append-only original-name label");
    await check('JSON.stringify(state.current.document.shapes.at(-1).points.map(position=>position.map(Math.round)))==="[[100,100],[350,100],[350,300],[100,300]]"', "Reverse clicks normalize rectangle corners");
    await box();
    await cdp("Input.insertText", {text:"rade_custom"});
    await press("Enter", "Enter");
    await waitFor('!element("labelPicker").open');
    await check('state.project.classes.length===3 && state.current.document.shapes.length===3', "Reusing a name does not duplicate class IDs");
    const first = await evaluate("state.current.item.id");
    await evaluate('window.testApi=api;api=async(path,method,...args)=>{if(method==="PUT")throw new Error("simulated save failure");return testApi(path,method,...args);};');
    await press("e", "KeyE");
    await waitFor('!state.busy');
    await check(`state.current.item.id===${JSON.stringify(first)} && state.dirty`, "Failed E save leaves the current image and edits in place");
    await evaluate('api=testApi;');
    await press("e", "KeyE");
    await waitFor(`state.current.item.id!==${JSON.stringify(first)} && !state.busy`);
    await check(`(async()=>{const saved=await api(projectUrl("/items/${first}"));return saved.status==="reviewed" && saved.document.shapes.length===3;})()`, "E persists annotations as reviewed and advances");
    const second = await evaluate("state.current.item.id");
    await cdp("Input.dispatchKeyEvent", {type:"keyDown", key:"e", code:"KeyE", autoRepeat:true});
    await check(`state.current.item.id===${JSON.stringify(second)} && state.current.version===0`, "Held E does not repeatedly review images");
    await press("e", "KeyE");
    await check(`state.current.item.id===${JSON.stringify(second)} && state.current.version===0`, "Empty-image review still requires explicit confirmation");
    await press("d", "KeyD");
    await waitFor(`state.current.item.id!==${JSON.stringify(second)} && !state.busy`);
    await check(`(async()=>{const saved=await api(projectUrl("/items/${second}"));return saved.status!=="reviewed";})()`, "D navigates without review");
    await press("a", "KeyA");
    await waitFor(`state.current.item.id===${JSON.stringify(second)} && !state.busy`);
    await press("a", "KeyA");
    await waitFor(`state.current.item.id===${JSON.stringify(first)} && !state.busy`);
    await check('state.current.document.shapes.length===3 && state.current.document.shapes.at(-1).label==="rade_custom"', "Saved rectangles and custom labels survive returning to the image");
    for (const width of [1600,1024,760]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width, height:1000, deviceScaleFactor:1, mobile:false});
      await sleep(100);
      await box([580,420], [630,470]);
      await check('(()=>{const rect=element("labelPicker").getBoundingClientRect();return rect.left>=0 && rect.top>=0 && rect.right<=innerWidth && rect.bottom<=innerHeight;})()', `Label picker stays on screen at width ${width}`);
      await click("#cancelPickedLabel");
      await waitFor('!element("labelPicker").open');
    }
    await click("#focusButton");
    await sleep(100);
    await box();
    await check('document.body.classList.contains("focus-mode") && element("labelPicker").open', "Label picker also works in the large-canvas workspace");
    await click("#cancelPickedLabel");
    await click("#focusButton");
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await sleep(100);
    const annotationBefore = await evaluate('JSON.stringify(state.current)');
    await click("#modelSettingsSmall");
    await click("#browseModelFile");
    await waitFor('modelBrowser.listing && !modelBrowser.loading');
    await check('document.querySelectorAll("[data-model-file]").length===1 && document.querySelectorAll("[data-model-directory]").length===1', "File browser lists folders and .pt files only");
    await click("#cancelModelBrowser");
    await check(`element("modelPath").value===${JSON.stringify(firstModel)} && state.project.model_path===${JSON.stringify(firstModel)}`, "Cancel leaves the saved model unchanged");
    await click("#browseModelFolder");
    await waitFor('modelBrowser.listing && !modelBrowser.loading');
    await click("#searchModelFolder");
    await waitFor('modelBrowser.listing.search && !modelBrowser.loading');
    await check('document.querySelectorAll("[data-model-file]").length===2', "Folder selection finds nested model candidates");
    const modelBrowserScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"model-browser.png"), Buffer.from(modelBrowserScreenshot.data,"base64"));
    const secondSelector = '[data-model-file='+JSON.stringify(secondModel)+']';
    await click(secondSelector);
    await check(`!element("modelBrowserDialog").open && element("modelPath").value===${JSON.stringify(secondModel)} && state.project.model_path===${JSON.stringify(firstModel)}`, "Selecting a candidate fills its absolute path without saving or loading it");
    await click("#saveModel");
    await waitFor(`state.project.model_path===${JSON.stringify(secondModel)}`);
    assert.equal(JSON.parse(fs.readFileSync(path.join(home,"projects.json"),"utf8"))[0].model_path, secondModel);
    context.diagnostic("Chosen model path persists in the isolated project configuration");
    await click("#browseModelFile");
    await waitFor('modelBrowser.listing && !modelBrowser.loading');
    await check(`modelBrowser.listing.path===${JSON.stringify(weightFolder)}`, "Reopening starts in the saved model's folder");
    await click("#modelBrowserUp");
    await waitFor(`modelBrowser.listing.path===${JSON.stringify(path.dirname(weightFolder))} && !modelBrowser.loading`);
    await click("#modelBrowserUp");
    await waitFor(`modelBrowser.listing.path===${JSON.stringify(modelFolder)} && !modelBrowser.loading`);
    await evaluate('element("modelBrowserPath").value='+JSON.stringify(path.join(home,"missing")));
    await click("#modelBrowserGo");
    await waitFor('!modelBrowser.loading && !!element("modelBrowserError").textContent');
    await check(`state.project.model_path===${JSON.stringify(secondModel)}`, "Invalid folders show an error and do not change the selected model");
    await evaluate('element("modelBrowserPath").value='+JSON.stringify(modelFolder));
    await click("#modelBrowserGo");
    await waitFor('!modelBrowser.loading && !element("modelBrowserError").textContent');
    await click('[data-model-file='+JSON.stringify(firstModel)+']');
    await click("#saveModel");
    await waitFor(`state.project.model_path===${JSON.stringify(firstModel)}`);
    await check(`JSON.stringify(state.current)===${JSON.stringify(annotationBefore)}`, "Switching models never replaces or edits the current annotations");
    await click("#browseModelFolder");
    await waitFor('modelBrowser.listing && !modelBrowser.loading');
    for (const width of [1024,760]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width, height:1000, deviceScaleFactor:1, mobile:false});
      await sleep(100);
      await check('(()=>{const rect=element("modelBrowserDialog").getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight;})()', `Model browser fits the viewport at width ${width}`);
    }
    await click("#cancelModelBrowser");
    assert.deepEqual(errors, [], "No browser runtime errors");
  } finally {
    if (socket?.readyState === WebSocket.OPEN && cdp) await cdp("Browser.close").catch(() => {});
    socket?.close();
    browser?.kill();
    server.kill();
    context.diagnostic(`Isolated fixtures: ${home}`);
  }
});
