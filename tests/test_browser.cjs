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

test("annotation, priority review, cleaning, split export and training controls", {timeout:120000}, async context => {
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
  const fixture = spawnSync(python, ["-c", "from PIL import Image; import sys; from pathlib import Path; [Image.new('RGB',(640,480),color).save(Path(sys.argv[1])/f'image_{index}.png') for index,color in enumerate(('#555577','#555578','#555579','#555579'))]", images], {windowsHide:true, encoding:"utf8"});
  assert.equal(fixture.status, 0, fixture.stderr || fixture.error?.message);
  const manifestPath = path.join(home, "capture-manifest.json");
  fs.writeFileSync(manifestPath, JSON.stringify({tasks:["capture-a", "capture-a", "capture-b", "capture-c"].map((group, index) => ({
    package_image:`image_${index}.png`, source_group:group
  }))}));
  fs.writeFileSync(path.join(home, "projects.json"), JSON.stringify([{id:"test", name:"Browser test", image_root:images,
    classes:["class_a", "class_b"], model_path:firstModel, manifest_path:manifestPath}]));
  const listener = net.createServer();
  await new Promise(resolve => listener.listen(0, "127.0.0.1", resolve));
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const url = `http://127.0.0.1:${port}`;
  const serverCode = `
import sys
import uvicorn
from fastapi.responses import HTMLResponse, Response
from labelpilot.server import create_app
app = create_app()
@app.middleware("http")
async def legacy_assets(request, call_next):
    if request.url.path == "/__legacy_cache_test":
        return HTMLResponse('<link rel="stylesheet" href="/static/style.css"><script src="/static/app.js" defer></script><body>Legacy assets</body>', headers={"Cache-Control": "no-store"})
    if not request.url.query and request.url.path in ("/static/app.js", "/static/style.css"):
        script = request.url.path.endswith(".js")
        return Response("window.legacyAssetLoaded=true;" if script else "body{--legacy-style-loaded:1}",
                        media_type="application/javascript" if script else "text/css",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})
    return await call_next(request)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
`;
  const server = spawn(python, ["-c", serverCode, String(port)], {
    cwd:root, windowsHide:true, env:{...process.env, LABELPILOT_HOME:home, YOLO_CONFIG_DIR:path.join(home,"ultralytics"), MPLCONFIGDIR:path.join(home,"matplotlib")}, stdio:"ignore"
  });
  let browser;
  let socket;
  let cdp;
  const errors = [];
  try {
    await eventually(async () => {try {return (await fetch(`${url}/api/health`)).ok;} catch {return false;}});
    browser = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--window-size=1600,1000", `${url}/__legacy_cache_test`], {windowsHide:true, stdio:"ignore"});
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
    const press = async (key, code, options={}) => {
      const windowsVirtualKeyCode = {Escape:27, Enter:13, Tab:9}[key] || key.toUpperCase().charCodeAt(0);
      await cdp("Input.dispatchKeyEvent", {type:"keyDown", key, code, windowsVirtualKeyCode, ...(key === "Enter" ? {text:"\r"} : {}), ...options});
      await cdp("Input.dispatchKeyEvent", {type:"keyUp", key, code, windowsVirtualKeyCode});
    };
    const mouseClick = async position => {
      await cdp("Input.dispatchMouseEvent", {type:"mousePressed", button:"left", clickCount:1, ...position});
      await cdp("Input.dispatchMouseEvent", {type:"mouseReleased", button:"left", clickCount:1, ...position});
    };
    const click = async selector => mouseClick(await evaluate(`(()=>{const rect=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};})()`));
    const point = coordinates => evaluate(`(()=>{const position=screenPoint(${JSON.stringify(coordinates)});const rect=canvas.getBoundingClientRect();return {x:rect.left+position[0],y:rect.top+position[1]};})()`);
    const checkCrosshair = async (position, message) => check(`(()=>{
      const guide=element("canvasCrosshair"),rect=guide.getBoundingClientRect(),bounds=canvas.getBoundingClientRect();
      const horizontal=getComputedStyle(guide,"::before"),vertical=getComputedStyle(guide,"::after");
      return !guide.hidden && Math.abs(rect.left-bounds.left)<1 && Math.abs(rect.top-bounds.top)<1 &&
        Math.abs(rect.right-bounds.right)<1 && Math.abs(rect.bottom-bounds.bottom)<1 &&
        horizontal.borderTopStyle==="dashed" && vertical.borderLeftStyle==="dashed" &&
        Math.abs(parseFloat(horizontal.left))<1 && Math.abs(parseFloat(horizontal.width)-bounds.width)<1 &&
        Math.abs(parseFloat(vertical.top))<1 && Math.abs(parseFloat(vertical.height)-bounds.height)<1 &&
        Math.abs(rect.top+parseFloat(horizontal.top)-${position.y})<1 &&
        Math.abs(rect.left+parseFloat(vertical.left)-${position.x})<1 &&
        getComputedStyle(guide).pointerEvents==="none" && document.elementFromPoint(${position.x},${position.y})===canvas;
    })()`, message);
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
    await eventually(()=>evaluate('window.legacyAssetLoaded===true && getComputedStyle(document.body).getPropertyValue("--legacy-style-loaded").trim()==="1"'));
    context.diagnostic("Browser starts with long-lived cached legacy JavaScript and CSS on the same origin");
    await cdp("Page.navigate", {url});
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await waitFor('typeof state!=="undefined" && state.current && !state.busy');
    await check('window.legacyAssetLoaded!==true && getComputedStyle(document.body).getPropertyValue("--legacy-style-loaded").trim()!=="1" && /[?&]v=[a-f0-9]{16}/.test(document.querySelector("script[src]").src) && /[?&]v=[a-f0-9]{16}/.test(document.querySelector("link[rel=stylesheet]").href)', "A normal navigation bypasses cached legacy assets and loads a matching versioned script and stylesheet");
    let cursor=await point([320,240]);
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...cursor});
    await check('element("canvasCrosshair").hidden', "Selection mode does not show rectangle crosshair guides");
    await press("r", "KeyR");
    await checkCrosshair(cursor, "R immediately draws four dashed arms to the canvas edges without moving the mouse");
    for (const pixelRatio of [1,2]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:pixelRatio, mobile:false});
      await sleep(100);
      cursor=await point([290,210]);
      await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...cursor});
      await cdp("Input.dispatchMouseEvent", {type:"mouseWheel", deltaX:0, deltaY:-120, ...cursor});
      await checkCrosshair(cursor, `Guides stay aligned and reach the canvas edges after zooming at pixel ratio ${pixelRatio}`);
    }
    const previousPan=await evaluate('({x:state.panX,y:state.panY})');
    await cdp("Input.dispatchKeyEvent", {type:"keyDown", key:" ", code:"Space", windowsVirtualKeyCode:32});
    await check('element("canvasCrosshair").hidden', "Holding Space hides rectangle guides while panning");
    await cdp("Input.dispatchMouseEvent", {type:"mousePressed", button:"left", clickCount:1, ...cursor});
    cursor={x:cursor.x+24,y:cursor.y+18};
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:1, ...cursor});
    await cdp("Input.dispatchMouseEvent", {type:"mouseReleased", button:"left", clickCount:1, ...cursor});
    await cdp("Input.dispatchKeyEvent", {type:"keyUp", key:" ", code:"Space", windowsVirtualKeyCode:32});
    await check(`Math.abs(state.panX-(${previousPan.x})-24)<1 && Math.abs(state.panY-(${previousPan.y})-18)<1`, "Space-drag still pans the image");
    await checkCrosshair(cursor, "Guides resume at the cursor after image panning");
    await evaluate('setBusy(true);');
    await check('element("canvasCrosshair").hidden', "Busy operations hide the visual-only guides");
    await evaluate('setBusy(false);');
    await checkCrosshair(cursor, "Guides resume after a busy operation without editing annotations");
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, x:5, y:5});
    await check('element("canvasCrosshair").hidden', "Leaving the canvas removes the guides");
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...cursor});
    await checkCrosshair(cursor, "Re-entering the canvas restores the guides");
    await press("v", "KeyV");
    await check('element("canvasCrosshair").hidden', "Switching to V hides the guides immediately");
    await press("p", "KeyP");
    await check('element("canvasCrosshair").hidden', "Polygon mode is not affected by the rectangle guides");
    await press("r", "KeyR");
    await evaluate('canvas.dispatchEvent(new PointerEvent("pointercancel",{pointerId:1}));');
    await check('element("canvasCrosshair").hidden', "Cancelled pointer interactions clear the guides");
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...cursor});
    await evaluate('window.dispatchEvent(new Event("blur"));');
    await check('element("canvasCrosshair").hidden', "Window blur does not leave a stale crosshair");
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await sleep(100);
    await press("f", "KeyF");
    cursor=await point([320,240]);
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...cursor});
    await checkCrosshair(cursor, "Guide geometry remains correct after restoring the image view");
    await check('!state.dirty && state.current.version===0 && state.current.document.shapes.length===0 && !state.rectangle', "Guide movements never create annotations or change the saved version");
    const crosshairScreenshot=await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"rectangle-crosshair.png"),Buffer.from(crosshairScreenshot.data,"base64"));
    await mouseClick(await point([100,100]));
    await checkCrosshair(await point([100,100]), "Guides stay visible after the first rectangle click");
    await check('!!state.rectangle && !state.dirty && state.current.document.shapes.length===0', "First click remains active after releasing the mouse");
    await mouseClick(await point([100,100]));
    await check('!!state.rectangle && !element("labelPicker").open', "Repeated clicks at the same point do not create an empty rectangle");
    await cdp("Input.dispatchMouseEvent", {type:"mouseMoved", buttons:0, ...await point([350,300])});
    await check('Math.abs(state.rectangle.end[0]-350)<1', "Rectangle follows the pointer without holding a button");
    await press("e", "KeyE");
    await check('!!state.rectangle && state.current.version===0', "E cannot review an unfinished rectangle");
    await mouseClick(await point([350,300]));
    await waitFor('element("labelPicker").open');
    await check('element("canvasCrosshair").hidden', "The completed rectangle label picker hides the guides");
    await check('state.current.document.shapes.length===0 && !state.dirty', "Pending rectangle is not saved before a label is chosen");
    await click('[data-label="class_a"]');
    await waitFor('!element("labelPicker").open');
    await check('state.current.document.shapes.length===1 && state.current.document.shapes[0].label==="class_a"', "Existing original label applies on click");
    await check('element("canvasCrosshair").hidden && !JSON.stringify(state.current.document).includes("canvasPointer")', "Crosshair guides are not part of the annotation document");
    await click("#undoButton");
    await check('state.current.document.shapes.length===0', "New rectangle is one undo operation");
    await click("#redoButton");
    await box();
    await check('document.activeElement.id==="confirmPickedLabel"', "Rectangle picker initially focuses confirmation so Q is immediately available");
    await press("q", "KeyQ", {modifiers:2});
    await press("q", "KeyQ", {autoRepeat:true});
    await evaluate('document.activeElement.dispatchEvent(new KeyboardEvent("keydown",{key:"q",code:"KeyQ",bubbles:true,isComposing:true}));');
    await check('element("labelPicker").open && state.current.document.shapes.length===1', "Modified, held and composing Q do not confirm the picker");
    await press("q", "KeyQ");
    await waitFor('!element("labelPicker").open');
    await check('state.current.document.shapes.length===2 && state.current.document.shapes.at(-1).label==="class_a" && state.current.version===0', "Q confirms the current label without saving, reviewing or advancing");
    await click("#undoButton");
    await box();
    await evaluate('document.querySelector("[data-label=class_b]").focus();');
    await press("Q", "KeyQ", {modifiers:8});
    await waitFor('!element("labelPicker").open');
    await check('state.current.document.shapes.length===2 && state.current.document.shapes.at(-1).label==="class_b"', "Uppercase Q activates the focused label button just like Enter");
    await click("#undoButton");
    await box();
    await evaluate('element("cancelPickedLabel").focus();');
    await press("q", "KeyQ");
    await check('!element("labelPicker").open && !state.pendingShape && state.current.document.shapes.length===1', "Q respects a focused cancel button instead of confirming another label");
    for (const key of ["q", "Enter"]) {
      await press("p", "KeyP");
      for (const coordinates of [[100,100],[350,100],[350,300]]) await mouseClick(await point(coordinates));
      await press("q", "KeyQ", {modifiers:2});
      await press("q", "KeyQ", {autoRepeat:true});
      await evaluate('canvas.dispatchEvent(new KeyboardEvent("keydown",{key:"q",code:"KeyQ",bubbles:true,isComposing:true}));');
      await check('state.polygon.length===3 && state.current.document.shapes.length===1', "Modified, held and composing Q leave the polygon unfinished");
      await press(key, key === "q" ? "KeyQ" : "Enter");
      await check('state.polygon.length===0 && state.current.document.shapes.length===2 && state.current.document.shapes.at(-1).shape_type==="polygon" && state.current.version===0', `${key} finishes the polygon without saving or reviewing`);
      await click("#undoButton");
    }
    await press("p", "KeyP");
    await mouseClick(await point([100,100]));
    await mouseClick(await point([350,100]));
    await press("q", "KeyQ");
    await check('state.polygon.length===2 && state.current.document.shapes.length===1', "Q cannot finish a polygon with fewer than three vertices");
    await press("Escape", "Escape");
    await box([350,300], [100,100]);
    await press("Escape", "Escape");
    await check('!state.pendingShape && !element("labelPicker").open && state.current.document.shapes.length===1', "Escape cancels the pending rectangle without changing existing objects");
    await box([350,300], [100,100]);
    await click("#pickerLabelInput");
    await press("q", "KeyQ", {text:"q"});
    await cdp("Input.insertText", {text:"rade_custom"});
    await press("Q", "KeyQ", {text:"Q",modifiers:8});
    await check('element("labelPicker").open && element("pickerLabelInput").value==="qrade_customQ" && state.current.document.shapes.length===1', "Lowercase and uppercase Q type literal label text without confirming");
    await press("e", "KeyE");
    await check('element("labelPicker").open && state.current.version===0', "E does not review while entering a label");
    await evaluate('window.testApi=api;api=async(path,method,...args)=>{if(method==="POST")throw new Error("simulated label failure");return testApi(path,method,...args);};');
    await evaluate('element("confirmPickedLabel").focus();');
    await press("q", "KeyQ");
    await check('!!state.pendingShape && element("labelPicker").open && element("pickerLabelError").textContent.includes("simulated label failure") && state.current.document.shapes.length===1', "Failed label creation preserves the pending rectangle for retry");
    await evaluate('api=testApi;');
    await press("Enter", "Enter");
    await waitFor('!element("labelPicker").open');
    await check('state.project.classes.length===3 && state.project.classes[2].name==="qrade_customQ" && state.current.document.shapes.at(-1).label==="qrade_customQ"', "Enter creates an append-only original-name label including literal q and Q");
    await check('JSON.stringify(state.current.document.shapes.at(-1).points.map(position=>position.map(Math.round)))==="[[100,100],[350,100],[350,300],[100,300]]"', "Reverse clicks normalize rectangle corners");
    await box();
    await click("#pickerLabelInput");
    await cdp("Input.insertText", {text:"qrade_customQ"});
    await evaluate('element("confirmPickedLabel").focus();');
    await press("q", "KeyQ");
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
    await check('state.current.document.shapes.length===3 && state.current.document.shapes.at(-1).label==="qrade_customQ"', "Saved rectangles and custom labels survive returning to the image");
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
    await evaluate('element("modelDialog").close();');
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await evaluate(`(async()=>{
      for(const item of state.items.slice(0,2)){
        const url=projectUrl("/items/"+item.id);
        const current=await api(url);
        await api(url,"PUT",{...current,expected_version:current.version,status:"draft"});
      }
    })()`);
    await click("#reviewQueueButton");
    await waitFor('state.current?.item.photo_id==="image_1" && !state.busy');
    await check('state.items.length===2 && element("imageList").textContent.includes("待确认空图")', "Priority review puts an unconfirmed empty draft ahead of a populated draft");
    await press("d", "KeyD");
    await waitFor('state.current?.item.photo_id==="image_0" && !state.busy');
    await press("e", "KeyE");
    await waitFor('state.items.length===1 && state.current?.item.photo_id==="image_1" && !state.busy');
    await click("#reviewQueueButton");
    await waitFor('state.items.length===4 && state.current?.item.photo_id==="image_0" && !state.busy');
    const downloads = path.join(home, "downloads");
    fs.mkdirSync(downloads);
    await cdp("Browser.setDownloadBehavior", {behavior:"allow", downloadPath:downloads});
    await click("#exportButton");
    await waitFor('element("exportDialog").open && element("exportPreviewStatus").textContent.includes("可用分组")');
    await check('element("splitDataset").checked && element("splitTrain").value==="80" && element("splitVal").value==="10" && element("splitTest").value==="10" && element("exportSplitUnit").value==="source_group"', "Projects with capture groups default to grouped 80/10/10 splitting");
    const setExportValue = (id, value) => evaluate(`element(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};element(${JSON.stringify(id)}).dispatchEvent(new Event("input",{bubbles:true}));`);
    await evaluate('window.exportApi=api;window.exportRequests=[];api=async(path,method,body)=>{if(path.endsWith("/export"))exportRequests.push(body);const result=await exportApi(path,method,body);if(path.endsWith("/export"))window.lastExportResult=result;return result;};');
    await setExportValue("splitTrain", 70);
    await check('element("createExport").disabled && element("exportSplitSummary").textContent.includes("90%")', "Invalid total disables export and shows the actual percentage");
    await evaluate('exportDataset()');
    await check('exportRequests.length===0', "Invalid ratios never reach the export API");
    await setExportValue("splitTrain", "");
    await check('element("createExport").disabled', "Empty percentage is rejected instead of silently becoming zero");
    await setExportValue("splitTrain", 50);
    await setExportValue("splitVal", 25);
    await setExportValue("splitTest", 25);
    await setExportValue("splitSeed", 1.5);
    await check('element("createExport").disabled', "Fractional seeds are rejected");
    await setExportValue("splitSeed", 42);
    await waitFor('element("exportPreviewStatus").textContent.includes("可用分组") && element("createExport").disabled');
    await check('exportRequests.length===0', "Too few capture groups block export before creating a ZIP");
    await evaluate('(async()=>{for(const row of state.items){const current=await api(projectUrl("/items/"+row.id));if(current.status!=="reviewed")await api(projectUrl("/items/"+row.id),"PUT",{...current,expected_version:current.version,status:"reviewed",confirmed_empty:true});}})()');
    await click("#refreshExportPreview");
    await waitFor('!element("createExport").disabled && element("exportImageCount").textContent==="4"');
    await check('element("exportEmptyCount").textContent==="3" && element("exportObjectCount").textContent==="3" && element("exportClassRows").textContent.includes("qrade_customQ") && element("exportPreviewIssues").textContent.includes("训练集中没有标注")', "Preview reports confirmed negatives, class counts and missing training classes");
    const beforeExport = await evaluate('JSON.stringify(state.current)');
    await evaluate('(async()=>{const route=projectUrl("/items/"+state.items.at(-1).id);const row=await api(route);await api(route,"PUT",{...row,expected_version:row.version,status:"reviewed",confirmed_empty:true});})()');
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("数据已变化") && !element("createExport").disabled');
    assert.equal(fs.readdirSync(downloads).length, 0, "A stale preview never downloads a ZIP");
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("训练 2 张 / 验证 1 张 / 测试 1 张") && !element("createExport").disabled');
    await check('JSON.stringify(exportRequests.at(-1).split)===JSON.stringify({train:50,val:25,test:25,seed:42,unit:"source_group"}) && lastExportResult.images===4', "The refreshed preview supplies its exact grouped split to the export API");
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 1);
    const splitExport = path.join(downloads, fs.readdirSync(downloads).find(name => name.endsWith(".zip")));
    const exportCheck = spawnSync(python, ["-c", `
import json, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    manifest = json.loads(archive.read('export_manifest.json'))
    assert manifest['split_counts'] == {'train': 2, 'val': 1, 'test': 1}
    assert manifest['split'] == 'grouped_by_source'
    assert all(entry['image'] in archive.namelist() for entry in manifest['items'])
    assert {entry['split'] for entry in manifest['items'] if entry['split_group'] == 'capture-a'} == {'train'}
    assert 'val: images/val' in archive.read('dataset.yaml').decode()
`, splitExport], {windowsHide:true, encoding:"utf8"});
    assert.equal(exportCheck.status, 0, exportCheck.stderr || exportCheck.error?.message);
    for (const [width, height] of [[1600,1000], [760,800], [480,740], [360,640]]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor:1, mobile:false});
      await sleep(100);
      await check('(()=>{const dialog=element("exportDialog"),rect=dialog.getBoundingClientRect(),button=element("createExport").getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && button.bottom<=rect.bottom;})()', `Split export dialog and download action fit the viewport at ${width} by ${height}`);
      if (width === 1600 || width === 360) {
        const screenshot = await cdp("Page.captureScreenshot", {format:"png"});
        fs.writeFileSync(path.join(home, width === 1600 ? "split-export.png" : "split-export-mobile.png"), Buffer.from(screenshot.data, "base64"));
      }
    }
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600, height:1000, deviceScaleFactor:1, mobile:false});
    await click("#splitDataset");
    await waitFor('!element("createExport").disabled');
    await check('element("exportSplitSettings").disabled && !element("createExport").disabled', "Users can keep the original unsplit export behavior");
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("未划分数据集") && !element("createExport").disabled');
    await check(`exportRequests.at(-1).split===null && JSON.stringify(state.current)===${JSON.stringify(beforeExport)}`, "Exports never change current annotations or their review state");
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 2);
    context.diagnostic("Custom ratios, validation, actual ZIP download and unsplit export passed");
    await evaluate('element("exportDialog").close();');
    await click("#cleanButton");
    await waitFor('!element("exportCleaning").hidden && !element("createExport").disabled');
    await check('element("exportImageCount").textContent==="3" && element("exportCleaningSummary").textContent.includes("重复副本 1 张")', "One-click cleaning previews identical duplicate removal");
    await click("#createExport");
    await waitFor('exportRequests.at(-1).clean===true && lastExportResult.images===3');
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 3);
    const cleaningScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"cleaning.png"), Buffer.from(cleaningScreenshot.data,"base64"));
    await evaluate('element("exportDialog").close();');

    await evaluate(`window.requestedTraining=null;
      api=async (route,method,body)=>{
        if(!route.endsWith("/trainings"))return exportApi(route,method,body);
        if(method==="POST"){window.requestedTraining=body;return {};}
        return window.requestedTraining ? [{id:"demo",status:"completed",created_at:"2026-11-18T08:00:00Z",
          progress:{epoch:1,epochs:1,device:"CPU",metrics:{map50_95:0.51,recall:0.62}},options:window.requestedTraining,
          split_counts:{train:2,val:1},cleaned_out:1,weights:${JSON.stringify(secondModel)},has_log:true,has_dataset:true}] : [];
      };`);
    await click("#trainingButton");
    await waitFor('element("trainingDialog").open && element("trainingRuns").textContent.includes("暂无")');
    await check('element("trainingUnit").value==="source_group" && element("trainingMode").value==="fine_tune"', "Training uses capture groups and the local project model by default");
    await evaluate('element("trainingMode").value="scratch"; element("trainingEpochs").value="1";');
    await click("#startTraining");
    await waitFor('element("trainingRuns").textContent.includes("mAP50–95 0.510")');
    await check('requestedTraining.epochs===1 && requestedTraining.mode==="scratch" && !!element("trainingRuns").querySelector("a[download][href$=weights]")', "Training controls submit settings and show metrics and the checkpoint link");
    const trainingScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"training.png"), Buffer.from(trainingScreenshot.data,"base64"));
    await click('[data-use-training="demo"]');
    await waitFor(`state.project.model_path===${JSON.stringify(secondModel)}`);
    await evaluate('element("trainingDialog").close(); api=exportApi;');
    assert.deepEqual(errors, [], "No browser runtime errors");
  } finally {
    if (socket?.readyState === WebSocket.OPEN && cdp) await cdp("Browser.close").catch(() => {});
    socket?.close();
    browser?.kill();
    server.kill();
    context.diagnostic(`Isolated fixtures: ${home}`);
  }
});
