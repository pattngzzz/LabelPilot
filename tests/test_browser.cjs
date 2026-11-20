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

test("annotation, review, export, training and workbench efficiency workflows", {timeout:120000}, async context => {
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
  const selectionHome = fs.mkdtempSync(path.join(os.tmpdir(), "labelpilot-image-selection-"));
  const selectionFolder = path.join(selectionHome, "待标注图片");
  fs.cpSync(images, selectionFolder, {recursive:true});
  fs.mkdirSync(path.join(selectionFolder, "子目录"));
  fs.copyFileSync(path.join(images,"image_0.png"), path.join(selectionFolder,"子目录","嵌套.PNG"));
  fs.writeFileSync(path.join(selectionFolder,"ignore.txt"), "not an image");
  const efficiencyFrames = path.join(selectionHome, "连续帧");
  fs.mkdirSync(efficiencyFrames);
  const efficiencyFixture = spawnSync(python, ["-c", `
from PIL import Image, ImageDraw
from pathlib import Path
import sys
directory = Path(sys.argv[1])
for index in range(6):
    image = Image.new("RGB", (320, 220), "#13202f" if index < 4 else "#5c2178")
    x = 35 + min(index, 4)*22
    ImageDraw.Draw(image).rectangle((x, 50, x+79, 116), fill="#edc13b")
    image.save(directory / f"part_{index:03}.png", "PNG" if index < 5 else "JPEG")
`, efficiencyFrames], {windowsHide:true, encoding:"utf8"});
  assert.equal(efficiencyFixture.status, 0, efficiencyFixture.stderr || efficiencyFixture.error?.message);
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
from labelpilot import evaluation
from labelpilot.blindsets import read_snapshot, verified_manifest
from labelpilot.comparison_details import record_frame, write_details
from labelpilot.evaluation_worker import frozen_frames
from labelpilot.server import create_app
from labelpilot.store import atomic_json, read_json

def comparison_fixture(run, stop, module, directory, copies):
    snapshot = read_snapshot(directory)
    manifest = verified_manifest(directory, snapshot)
    setup = read_json(run / "compare.json")
    frames = frozen_frames(directory, snapshot, manifest)
    negative = next(i for i, (_, _, labels, _) in enumerate(frames) if not labels)
    results = {}
    with write_details(run) as detail:
        for role in ("old", "new"):
            counts = {name: dict.fromkeys(("ground_truth", "matched", "missed", "false_positives"), 0)
                      for name in manifest["classes"]}
            for i, (_, item, labels, size) in enumerate(frames):
                raw = [(name, .85, box) for name, box in labels if (name == "class_a") == (role == "old")]
                if role == "old" and i == negative:
                    raw.append(("class_b", .65, (10, 10, 100, 100)))
                predictions = evaluation.count_detections(counts, labels, raw, .5)
                record_frame(detail, role, i, item, labels, size, predictions)
            for row in counts.values():
                row["recall"] = row["matched"] / row["ground_truth"] if row["ground_truth"] else None
            results[role] = counts
    rows = [{"name": name, "old": results["old"][name], "new": results["new"][name]} for name in manifest["classes"]]
    atomic_json(run / "report.json", {"snapshot": snapshot, "classes": rows, "has_details": True, **setup})

evaluation.has_training_dependencies = lambda: True
evaluation.run_local_worker = comparison_fixture
app = create_app()
def audit_prediction_fixture(project_id, item_id, options, **kwargs):
    current = app.state.store.detail(project_id, item_id)
    shapes = current["document"]["shapes"]
    predictions = [{**shapes[0], "label": "another", "score": .92}] if current["item"]["photo_id"] == "part_002" and shapes else []
    return {"base_version": current["version"], "image_sha256": current["image_sha256"], "shapes": predictions,
            "provenance": {"confidence": options["confidence"]}}

app.state.inference.predict = audit_prediction_fixture
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
    let labelFileChooser = null;
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
      if (data.method === "Page.fileChooserOpened") labelFileChooser = data.params;
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
    const click = async selector => mouseClick(await evaluate(`(()=>{const node=document.querySelector(${JSON.stringify(selector)});node.scrollIntoView({block:"center"});const rect=node.getBoundingClientRect();return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};})()`));
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
    await evaluate('element("autoDraft").checked=false; element("autoDraft").dispatchEvent(new Event("change"));');
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
    await waitFor('exportAugmentation.loaded');
    await check('element("augmentationPreset").value==="none" && element("augmentationPreset").options.length===7 && element("augmentationCopies").disabled', "Export defaults to no augmentation and offers five presets plus all-presets mode");
    const setExportValue = (id, value) => evaluate(`element(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};element(${JSON.stringify(id)}).dispatchEvent(new Event(element(${JSON.stringify(id)}).tagName==="SELECT" ? "change" : "input",{bubbles:true}));`);
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
    await setExportValue("augmentationPreset", "gentle");
    await evaluate('element("augmentationPreset").dispatchEvent(new Event("change",{bubbles:true}));');
    await check('element("createExport").disabled && element("augmentationSummary").textContent.includes("先划分")', "Augmentation cannot export an unsplit dataset");
    const requestsBeforeInvalidAugmentation = await evaluate('exportRequests.length');
    await evaluate('exportDataset()');
    await check(`exportRequests.length===${requestsBeforeInvalidAugmentation}`, "Invalid augmentation settings never reach the export API");
    await click("#splitDataset");
    await click("#includeImages");
    await check('element("createExport").disabled && element("augmentationSummary").textContent.includes("包含图片")', "Augmentation requires matching image files");
    await click("#includeImages");
    await click("#reviewedOnly");
    await check('element("createExport").disabled && element("augmentationSummary").textContent.includes("已人工确认")', "Augmentation cannot amplify unreviewed annotations");
    await click("#reviewedOnly");
    for (const value of ["",0,4,1.5]) {
      await setExportValue("augmentationCopies", value);
      await check('element("createExport").disabled', `Invalid augmentation copy count ${JSON.stringify(value)} disables export`);
    }
    await setExportValue("augmentationCopies", 2);
    await setExportValue("augmentationSeed", 1.5);
    await check('element("createExport").disabled', "Fractional augmentation seed disables export");
    await setExportValue("augmentationSeed", 43);
    for (const preset of ["gentle","lighting","quality","rotation","horizontal"]) {
      await setExportValue("augmentationPreset", preset);
      await evaluate('element("augmentationPreset").dispatchEvent(new Event("change",{bubbles:true}));');
      await waitFor('!element("createExport").disabled');
      await check('element("augmentationDescription").textContent.length>20 && element("augmentationSummary").textContent.includes("3 倍")', `${preset} displays its limitations and added-copy summary`);
    }
    const previousDownloads = fs.readdirSync(downloads).filter(name => name.endsWith(".zip"));
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("训练增强 4 张") && !element("createExport").disabled');
    await check('lastExportResult.images===8 && lastExportResult.original_images===4 && lastExportResult.augmented_images===4 && JSON.stringify(exportRequests.at(-1).augmentation)===JSON.stringify({preset:"horizontal",copies:2,seed:43})', "Chosen augmentation reaches the backend and reports original plus augmented counts");
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 3);
    const augmentedExport = path.join(downloads, fs.readdirSync(downloads).find(name => name.endsWith(".zip") && !previousDownloads.includes(name)));
    const augmentationCheck = spawnSync(python, ["-c", "import io,json,sys,zipfile; from PIL import Image; archive=zipfile.ZipFile(sys.argv[1]); manifest=json.loads(archive.read('export_manifest.json')); assert manifest['split_counts']=={'train':6,'val':1,'test':1}; assert manifest['integrity_check']['passed'] and manifest['integrity_check']['augmented_images_checked']==4; assert manifest['integrity_check']['annotations_checked']==sum(item['annotation_count'] for item in manifest['items']); originals={item['id']:item for item in manifest['items'] if 'augmentation' not in item}; variants=[item for item in manifest['items'] if 'augmentation' in item]; assert len(originals)==len(variants)==4; assert len({item['image'] for item in manifest['items']})==8; assert all(item['split']==originals[item['source_id']]['split']=='train' and item['annotation_count']==originals[item['source_id']]['annotation_count'] for item in variants); assert all(not item['human_reviewed'] for item in variants); [Image.open(io.BytesIO(archive.read(item['image']))).load() for item in variants]; assert all(len(archive.read(item['image'].replace('images/','labels/',1).rsplit('.',1)[0]+'.txt').decode().splitlines())==item['annotation_count'] for item in manifest['items']); print('Augmented YOLO ZIP decoded; every image has verified box counts and matching provenance')", augmentedExport], {windowsHide:true, encoding:"utf8"});
    assert.equal(augmentationCheck.status, 0, augmentationCheck.stderr || augmentationCheck.error?.message);
    await check('element("exportResult").textContent.includes("标注完整性校验通过")', "Export displays the completed runtime image and object integrity checks");
    await check(`JSON.stringify(state.current)===${JSON.stringify(beforeExport)}`, "Augmented export never alters the current annotation or review state");
    await setExportValue("augmentationPreset", "all");
    await check('element("augmentationCopiesLabel").textContent.includes("每种方案") && element("augmentationSummary").textContent.includes("额外生成 10 张") && element("augmentationSummary").textContent.includes("11 倍")', "All-presets mode counts copies per preset and displays the true total");
    await setExportValue("augmentationCopies", 3);
    await check('element("augmentationSummary").textContent.includes("额外生成 15 张") && element("augmentationSummary").textContent.includes("16 倍")', "All-presets mode correctly counts the maximum copy setting");
    await setExportValue("augmentationCopies", 1);
    await check('element("augmentationSummary").textContent.includes("每方案 1 张") && element("augmentationSummary").textContent.includes("含原图共 6 张") && element("augmentationDescription").textContent.includes("不连续叠加")', "All-presets mode explains independent generation and the recommended count");
    await waitFor('!element("createExport").disabled');
    const downloadsBeforeAll = fs.readdirSync(downloads).filter(name => name.endsWith(".zip"));
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("训练增强 10 张") && !element("createExport").disabled');
    await check('lastExportResult.images===14 && lastExportResult.augmented_images===10 && lastExportResult.split_counts.train===12 && JSON.stringify(exportRequests.at(-1).augmentation)===JSON.stringify({preset:"all",copies:1,seed:43})', "All-presets selection reaches the backend with correct train-only totals");
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 4);
    const allExport = path.join(downloads, fs.readdirSync(downloads).find(name => name.endsWith(".zip") && !downloadsBeforeAll.includes(name)));
    const allCheck = spawnSync(python, ["-c", "import io,json,sys,zipfile; from PIL import Image; archive=zipfile.ZipFile(sys.argv[1]); manifest=json.loads(archive.read('export_manifest.json')); assert manifest['split_counts']=={'train':12,'val':1,'test':1}; assert manifest['augmentation']['preset']=='all' and manifest['augmentation']['variants_per_image']==5; assert manifest['integrity_check']['passed'] and manifest['integrity_check']['augmented_images_checked']==10; originals={item['id']:item for item in manifest['items'] if 'augmentation' not in item}; variants=[item for item in manifest['items'] if 'augmentation' in item]; assert len(originals)==4 and len(variants)==10; assert len({item['image'] for item in manifest['items']})==14; assert len(archive.namelist())==len(set(archive.namelist())); assert all({entry['augmentation']['preset'] for entry in variants if entry['source_id']==item['id']}==({'gentle','lighting','quality','horizontal','rotation'} if item['split']=='train' else set()) for item in originals.values()); assert all(item['split']==originals[item['source_id']]['split']=='train' and item['annotation_count']==originals[item['source_id']]['annotation_count'] for item in variants); [Image.open(io.BytesIO(archive.read(item['image']))).load() for item in variants]; assert all(len(archive.read(item['image'].replace('images/','labels/',1).rsplit('.',1)[0]+'.txt').decode().splitlines())==item['annotation_count'] for item in manifest['items']); print('All five presets independently downloaded with matching boxes and unique paths')", allExport], {windowsHide:true, encoding:"utf8"});
    assert.equal(allCheck.status, 0, allCheck.stderr || allCheck.error?.message);
    await check(`JSON.stringify(state.current)===${JSON.stringify(beforeExport)}`, "All-presets export preserves the current annotation and review state");
    await setExportValue("augmentationPreset", "rotation");
    await setExportValue("augmentationCopies", 3);
    await check('element("augmentationDescription").textContent.includes("90°") && element("augmentationDescription").textContent.includes("交换宽高") && element("augmentationSummary").textContent.includes("含原图共 4 张")', "Rotation explains its three angles, changed dimensions and independent copy count");
    const downloadsBeforeRotation = fs.readdirSync(downloads).filter(name => name.endsWith(".zip"));
    await waitFor('!element("createExport").disabled');
    await click("#createExport");
    await waitFor('element("exportResult").textContent.includes("训练增强 6 张") && !element("createExport").disabled');
    await check('lastExportResult.images===10 && lastExportResult.split_counts.train===8 && JSON.stringify(exportRequests.at(-1).augmentation)===JSON.stringify({preset:"rotation",copies:3,seed:43})', "Rotation exports three different angles for every training original without changing validation or test images");
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 5);
    const rotationExport = path.join(downloads, fs.readdirSync(downloads).find(name => name.endsWith(".zip") && !downloadsBeforeRotation.includes(name)));
    const rotationCheck = spawnSync(python, ["-c", `
import io,json,math,sys,zipfile
from PIL import Image
archive=zipfile.ZipFile(sys.argv[1])
manifest=json.loads(archive.read('export_manifest.json'))
assert manifest['split_counts']=={'train':8,'val':1,'test':1}
assert manifest['integrity_check']['passed'] and manifest['integrity_check']['augmented_images_checked']==6
originals={entry['id']:entry for entry in manifest['items'] if 'augmentation' not in entry}
variants=[entry for entry in manifest['items'] if 'augmentation' in entry]
assert len(variants)==6 and all(entry['split']=='train' for entry in variants)
def labels(entry):
    name=entry['image'].replace('images/','labels/',1).rsplit('.',1)[0]+'.txt'
    return [line.split() for line in archive.read(name).decode().splitlines()]
for original in originals.values():
    angles={entry['augmentation']['operations']['rotation_degrees'] for entry in variants if entry['source_id']==original['id']}
    assert angles==({90,180,270} if original['split']=='train' else set())
for entry in variants:
    original=originals[entry['source_id']]
    angle=entry['augmentation']['operations']['rotation_degrees']
    with Image.open(io.BytesIO(archive.read(entry['image']))) as image:
        image.load()
        assert image.size==((480,640) if angle in (90,270) else (640,480))
    old,new=labels(original),labels(entry)
    assert len(old)==len(new)==entry['annotation_count']==original['annotation_count']
    for source,target in zip(old,new):
        assert source[0]==target[0]
        center_x,center_y,width,height=map(float,source[1:])
        expected=([center_y,1-center_x,height,width] if angle==90 else
                  [1-center_x,1-center_y,width,height] if angle==180 else [1-center_y,center_x,height,width])
        assert all(math.isclose(actual,wanted,abs_tol=1e-10) for actual,wanted in zip(map(float,target[1:]),expected))
print('Rotation ZIP: all angles, dimensions, exact normalized boxes and train-only scope verified')
`, rotationExport], {windowsHide:true, encoding:"utf8"});
    assert.equal(rotationCheck.status, 0, rotationCheck.stderr || rotationCheck.error?.message);
    await check(`JSON.stringify(state.current)===${JSON.stringify(beforeExport)}`, "Rotating export copies never edits the original image annotations or review state");
    await setExportValue("augmentationCopies", 1);
    await setExportValue("augmentationPreset", "gentle");
    await check('!element("augmentationCopiesLabel").textContent.includes("每种方案") && element("augmentationSummary").textContent.includes("含原图共 2 张")', "Returning to a single preset restores its original count unit");
    await setExportValue("augmentationPreset", "all");
    for (const width of [1600,760,480]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:1000,deviceScaleFactor:1,mobile:false});
      await sleep(100);
      await evaluate('element("augmentationPreset").scrollIntoView({block:"center"});');
      await check('(()=>{const dialog=element("exportDialog");const rect=dialog.getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && element("augmentationPreset").clientWidth>100;})()', `Augmentation controls fit and remain scroll-accessible at width ${width}`);
    }
    const augmentationScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"augmentation-export.png"), Buffer.from(augmentationScreenshot.data,"base64"));
    context.diagnostic("Five augmentation presets and all-presets mode, copy totals, safety validation, downloaded YOLO ZIPs and annotation preservation passed");
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('element("exportDialog").close();');
    await click("#cleanButton");
    await waitFor('!element("exportCleaning").hidden && !element("createExport").disabled');
    await check('element("exportImageCount").textContent==="3" && element("exportCleaningSummary").textContent.includes("重复副本 1 张")', "One-click cleaning previews identical duplicate removal");
    await click("#createExport");
    await waitFor('exportRequests.at(-1).clean===true && lastExportResult.images===3');
    await eventually(() => fs.readdirSync(downloads).filter(name => name.endsWith(".zip")).length === 6);
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
    await click("#comparisonButton");
    await waitFor('element("comparisonDialog").open && element("comparisonSet").options[0]?.text==="暂无盲测集" && element("comparisonModels").options.length===1');
    await check(`element("startComparison").disabled && element("comparisonOldPath").value===${JSON.stringify(secondModel)}`, "Model comparison suggests the current weight and waits for a frozen set");
    await click("#openBlindset");
    await waitFor('element("blindsetDialog").open');
    await check('element("blindsetUnit").value==="source_group"', "The blind set defaults to the same capture grouping as training");
    await setExportValue("blindsetName", "月度盲测");
    await setExportValue("blindsetPercent", 100);
    await click("#freezeBlindset");
    await waitFor('!element("blindsetDialog").open && element("comparisonSet").options[0].text.includes("月度盲测")');
    await check('element("comparisonSetInfo").textContent.includes("3 张图") && !element("startComparison").disabled', "A real frozen snapshot is selectable and displays the cleaned image count");
    await click("#browseComparisonOld");
    await waitFor('modelBrowser.listing && !modelBrowser.loading');
    await setExportValue("modelBrowserPath", modelFolder);
    await click("#modelBrowserGo");
    await waitFor(`!modelBrowser.loading && modelBrowser.listing.path===${JSON.stringify(modelFolder)}`);
    await click('[data-model-file='+JSON.stringify(firstModel)+']');
    await check(`element("comparisonOldPath").value===${JSON.stringify(firstModel)} && element("modelPath").value===${JSON.stringify(secondModel)}`, "Selecting a comparison weight leaves the saved preannotation weight intact");
    await setExportValue("comparisonNewPath", secondModel);
    await click("#startComparison");
    await waitFor('!element("comparisonReport").hidden && element("comparisonClassRows").rows.length>1');
    await check('(()=>{const rows=[...element("comparisonClassRows").rows];return rows.some(row=>row.cells[1].textContent!=="0" && row.cells[2].textContent==="0%" && row.cells[5].textContent==="100%" && row.cells[7].textContent==="0") && rows.some(row=>row.cells[1].textContent==="0" && row.cells[2].textContent==="—" && row.cells[4].textContent==="1");})()', "The table shows both models' per-class recall, misses and false positives, plus the no-ground-truth case");
    await click("#downloadComparison");
    const comparisonDownload = await eventually(() => fs.readdirSync(downloads).find(name => /^labelpilot-compare-.+\.json$/.test(name)));
    const comparisonReport = JSON.parse(fs.readFileSync(path.join(downloads, comparisonDownload)));
    assert.equal(comparisonReport.snapshot.name, "月度盲测");
    assert.equal(comparisonReport.options.match_iou, .5);
    const comparisonScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"comparison.png"), Buffer.from(comparisonScreenshot.data,"base64"));
    const setReviewValue = (id, value) => evaluate(`(()=>{const field=element(${JSON.stringify(id)});field.value=${JSON.stringify(value)};field.dispatchEvent(new Event("change",{bubbles:true}));})()`);
    await click('[data-review-label="class_a"][data-review-kind="new_missed"]');
    await waitFor('element("comparisonReviewDialog").open && !element("comparisonFrame").hidden && element("comparisonFrameNewImage").querySelector("img")?.naturalWidth');
    await check('element("comparisonReviewKind").value==="new_missed" && element("comparisonReviewClass").value==="class_a" && element("comparisonFrameOldImage").querySelectorAll(".comparison-frame-box.matched").length===1 && element("comparisonFrameNewImage").querySelector(".comparison-frame-box.truth").textContent.includes("漏检")', "A missed count opens the exact class and image with frozen truth and old/new predictions");
    await setReviewValue("comparisonReviewKind", "newly_missed");
    await waitFor('element("comparisonFrameNewImage").querySelector(".focused")?.textContent.includes("新增漏检")');
    await check('element("comparisonReviewStatus").textContent.includes("1 张图 · 1 个漏检")', "The regression filter highlights the specific newly missed target");
    const reviewScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"comparison-review.png"), Buffer.from(reviewScreenshot.data,"base64"));
    for (const width of [1024, 760, 400]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:850,deviceScaleFactor:1,mobile:false});
      await sleep(100);
      await check('(()=>{const dialog=element("comparisonReviewDialog");const rect=dialog.getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && element("comparisonFrameNewImage").querySelector("img").clientWidth>100;})()', `Per-image review stays usable at width ${width}`);
    }
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await setReviewValue("comparisonReviewKind", "fixed_missed");
    await waitFor('element("comparisonFrame").hidden && element("comparisonReviewStatus").textContent.includes("0 张图")');
    const fixedClass = await evaluate('comparisonUi.reportData.classes.find(row=>row.old.missed>0 && row.new.matched>0).name');
    await setReviewValue("comparisonReviewClass", fixedClass);
    await waitFor('element("comparisonFrameNewImage").querySelector(".focused")?.textContent.includes("修复漏检")');
    await evaluate('element("comparisonReviewDialog").close();');
    await click('[data-review-label="class_b"][data-review-kind="old_false_positives"]');
    await waitFor('element("comparisonFrameOldImage").querySelector(".comparison-frame-box.false-positive") && !element("comparisonFrame").hidden');
    await check('element("comparisonReviewKind").value==="old_false_positives" && element("comparisonReviewStatus").textContent.includes("1 个误检") && !element("comparisonFrameOldImage").querySelector(".comparison-frame-box.truth")', "The false-positive count opens the negative image and renders its unmatched prediction");
    await evaluate('element("comparisonReviewDialog").close();');
    for (const width of [1024, 760, 400]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:850,deviceScaleFactor:1,mobile:false});
      await sleep(100);
      await check('(()=>{const dialog=element("comparisonDialog");const rect=dialog.getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && element("comparisonOldPath").clientWidth>100;})()', `Model comparison remains usable at width ${width}`);
    }
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('element("comparisonDialog").close();');
    await click("#comparisonButton");
    await waitFor('!element("comparisonReport").hidden && element("comparisonRunStatus").textContent==="对比完成"');
    await evaluate('element("comparisonDialog").close();');
    const originalItemId = await evaluate('state.current.item.id');
    const originalStoredAnnotation = await evaluate('(async()=>JSON.stringify(await api(projectUrl("/items/"+state.current.item.id))))()');
    await click("#newProjectButton");
    await evaluate(`window.imageSelectionApi=api;window.nativeSelectionRequests=[];api=async(url,method,body)=>{
      if(url!=="/api/images/select")return imageSelectionApi(url,method,body);
      nativeSelectionRequests.push(body);
      return new Promise((resolve,reject)=>{window.finishNativeSelection=resolve;window.failNativeSelection=reject;});
    };`);
    await click("#chooseImageFiles");
    await waitFor('projectImages.selecting && nativeSelectionRequests.length===1');
    await check('nativeSelectionRequests[0].mode==="file" && element("chooseImageFiles").disabled && element("chooseImageFolder").disabled && element("newImageRoot").disabled && element("createProject").disabled && element("newImageSelection").textContent.includes("Windows")', "The button requests the native file picker and prevents repeat selection or import while it is open");
    await evaluate('selectLocalImages("folder");finishNativeSelection({cancelled:true});');
    await waitFor('!projectImages.selecting');
    await check(`nativeSelectionRequests.length===1 && state.project.id==="test" && JSON.stringify(state.current)===${JSON.stringify(beforeExport)} && projectImages.files.length===0 && element("newImageRoot").value===""`, "Cancelling the native picker leaves the project, annotation and previous paths unchanged");
    await click("#chooseImageFiles");
    await waitFor('projectImages.selecting && nativeSelectionRequests.length===2');
    await evaluate('failNativeSelection(new Error("Native picker fixture failure"));');
    await waitFor('!projectImages.selecting && [...document.querySelectorAll(".toast")].some(node=>node.textContent.includes("Native picker fixture failure"))');
    await check('projectImages.files.length===0 && !element("chooseImageFiles").disabled && !element("createProject").disabled', "Native window failures preserve selection and restore the controls with a visible error");
    await click("#chooseImageFiles");
    await waitFor('projectImages.selecting && nativeSelectionRequests.length===3');
    await evaluate(`finishNativeSelection({cancelled:false,image_root:${JSON.stringify(selectionFolder)},image_files:["image_0.png","image_1.png"]});`);
    await waitFor('!projectImages.selecting && projectImages.files.length===2');
    await check(`element("newImageRoot").value===${JSON.stringify(selectionFolder)} && element("newImageSelection").textContent.includes("仅导入已选 2 张") && !element("useEntireImageFolder").hidden`, "Native selection automatically fills the full Chinese path and explicit file subset without typing an address");
    await click("#chooseImageFolder");
    await waitFor('projectImages.selecting && nativeSelectionRequests.length===4');
    await check(`nativeSelectionRequests[3].mode==="folder" && nativeSelectionRequests[3].path===${JSON.stringify(selectionFolder)}`, "Folder selection starts at the previously selected local folder");
    await evaluate('finishNativeSelection({cancelled:true});');
    await waitFor('!projectImages.selecting');
    await check('projectImages.files.length===2', "Cancelling a later folder pick preserves the already confirmed file subset");
    for (const width of [1600,760,480]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:1000,deviceScaleFactor:1,mobile:false});
      await sleep(100);
      await check('(()=>{const dialog=element("projectDialog"),rect=dialog.getBoundingClientRect();return rect.left>=0 && rect.right<=innerWidth && rect.top>=0 && rect.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth;})()', `The project selection form fits the viewport at width ${width}`);
    }
    fs.writeFileSync(path.join(home,"image-picker.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data,"base64"));
    await evaluate('api=imageSelectionApi;');
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await setExportValue("newProjectName", "Selected images");
    await setExportValue("newClasses", "Raw_Label-v1\nclass_b");
    await click("#createProject");
    await waitFor('state.project.id!=="test" && state.project.total===2 && state.current && !state.busy && !element("projectDialog").open');
    const selectedProjectId = await evaluate('state.project.id');
    await check('state.items.length===2 && state.project.classes[0].name==="Raw_Label-v1" && state.current.width===640 && state.current.height===480', "Selected image files open in the annotation canvas with original custom labels");
    await press("r","KeyR");
    await mouseClick(await point([100,100]));
    await mouseClick(await point([240,240]));
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('!element("labelPicker").open && state.current.document.shapes.length===1');
    const selectedItemId = await evaluate('state.current.item.id');
    await press("e","KeyE");
    await waitFor(`state.current.item.id!==${JSON.stringify(selectedItemId)} && !state.busy`);
    await check(`(async()=>(await api(projectUrl("/items/"+${JSON.stringify(selectedItemId)}))).status==="reviewed")()`, "R, Q and E annotate, save and review newly chosen images through the real backend");
    fs.copyFileSync(path.join(images,"image_0.png"), path.join(selectionFolder,"later_added.png"));
    await click("#refreshButton");
    await waitFor('!state.busy && state.items.length===2');
    await evaluate('window.beforeImportReload=true;');
    await cdp("Page.reload");
    await waitFor(`window.beforeImportReload!==true && typeof state!=="undefined" && state.project?.id===${JSON.stringify(selectedProjectId)} && state.current && !state.busy`);
    await check(`(async()=>state.items.length===2 && (await api(projectUrl("/items/"+${JSON.stringify(selectedItemId)}))).status==="reviewed")()`, "Reload and rescan preserve the selected subset and saved annotations without adding other images");
    await click("#newProjectButton");
    await evaluate(`window.imageSelectionApi=api;api=async(url,method,body)=>url==="/api/images/select" ? {cancelled:false,image_root:${JSON.stringify(selectionFolder)},image_files:[]} : imageSelectionApi(url,method,body);`);
    await click("#chooseImageFolder");
    await waitFor('!projectImages.selecting && projectImages.files.length===0');
    await evaluate('api=imageSelectionApi;');
    await check('projectImages.files.length===0 && element("useEntireImageFolder").hidden', "Folder selection exits file-subset mode instead of retaining a hidden filter");
    await setExportValue("newProjectName", "Whole folder");
    await setExportValue("newClasses", "Raw_Label-v1\nclass_b");
    await click("#createProject");
    await waitFor(`state.project.id!==${JSON.stringify(selectedProjectId)} && state.project.total===6 && state.current && !state.busy`);
    await check('state.items.length===6 && state.items.some(item=>item.photo_id==="嵌套")', "Folder import includes nested images while excluding unsupported files");
    await check(`(async()=>JSON.stringify(await api("/api/projects/test/items/"+${JSON.stringify(originalItemId)}))===${JSON.stringify(originalStoredAnnotation)})()`, "Choosing new datasets never overwrites annotations in the original project");
    context.diagnostic("Stubbed native-picker responses verify button requests, cancellation, errors and automatic paths; real backend imports, persistence and annotation saving passed");
    const labelDownloads = path.join(home, "label-downloads");
    fs.mkdirSync(labelDownloads);
    await cdp("Browser.setDownloadBehavior", {behavior:"allow", downloadPath:labelDownloads});
    await check('!element("exportLabelsButton").disabled && element("exportLabelsButton").closest(".inspector")!==null', "A standalone label export button is available in the right inspector");
    await evaluate('setBusy(true);exportLabels();');
    await check('element("exportLabelsButton").disabled', "Label export is disabled while an operation is busy");
    await evaluate('setBusy(false);(()=>{const project=state.project;state.project=null;try{exportLabels();}finally{state.project=project;}})();');
    assert.equal(fs.readdirSync(labelDownloads).length, 0, "Busy or missing-project states cannot download labels");
    await click("#newLabelInput");
    await cdp("Input.insertText", {text:"Unused_Label-v2_样本"});
    await evaluate('window.beforeLabelExportApi=api;api=async(url,method,body)=>{if(method==="POST" && url.endsWith("/labels"))await new Promise(resolve=>{window.continueLabelCreation=resolve;});return beforeLabelExportApi(url,method,body);};');
    await click("#addLabelButton");
    await waitFor('element("addLabelButton").disabled && typeof continueLabelCreation==="function"');
    await evaluate('exportLabels();');
    await check('element("exportLabelsButton").disabled', "Pending label creation cannot export a stale class list");
    assert.equal(fs.readdirSync(labelDownloads).length, 0);
    await evaluate('continueLabelCreation();');
    await waitFor('state.project.classes.at(-1).name==="Unused_Label-v2_样本" && !element("exportLabelsButton").disabled');
    await evaluate('api=beforeLabelExportApi;');
    const labelExportState = await evaluate('JSON.stringify({current:state.current,dirty:state.dirty,activeClass:state.activeClass,classes:state.project.classes})');
    const labelProjectId = await evaluate('state.project.id');
    const expectedLabelText = await evaluate('state.project.classes.map(item=>item.name).join("\\n")+"\\n"');
    await click("#newLabelInput");
    await cdp("Input.insertText", {text:"not_submitted"});
    await click("#classesTab");
    for (const width of [1600,1024,760]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:1000,deviceScaleFactor:1,mobile:false});
      await sleep(100);
      await check('(()=>{const button=element("exportLabelsButton"),rect=button.getBoundingClientRect(),panel=button.closest(".inspector").getBoundingClientRect();return getComputedStyle(button).display!=="none" && rect.width>0 && rect.left>=panel.left && rect.right<=panel.right && !button.disabled;})()', `The label export button stays visible and fits the inspector at width ${width}`);
    }
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await click("#exportLabelsButton");
    const downloadedLabels = path.join(labelDownloads, "classes.txt");
    await eventually(() => fs.existsSync(downloadedLabels));
    assert.equal(fs.readFileSync(downloadedLabels, "utf8"), expectedLabelText, "Downloaded classes preserve exact UTF-8 names, appended unused labels and class ID order, excluding unsubmitted input");
    await check(`JSON.stringify({current:state.current,dirty:state.dirty,activeClass:state.activeClass,classes:state.project.classes})===${JSON.stringify(labelExportState)} && !element("exportDialog").open`, "Standalone label export never saves, reviews or changes image annotations or opens dataset export");
    assert.deepEqual(fs.readdirSync(labelDownloads), ["classes.txt"], "Standalone export downloads no image, annotation or ZIP files");
    const reusedLabels = await evaluate(`(async()=>await api("/api/projects","POST",{name:"Reuse exported labels",image_root:${JSON.stringify(selectionFolder)},class_map_path:${JSON.stringify(downloadedLabels)}}))()`);
    assert.equal(reusedLabels.classes.map(label=>label.name).join("\n")+"\n", expectedLabelText, "The downloaded classes.txt imports into a new project with identical names and IDs");
    const labelDownloadsAfterReload = path.join(home, "label-downloads-after-reload");
    fs.mkdirSync(labelDownloadsAfterReload);
    await cdp("Browser.setDownloadBehavior", {behavior:"allow", downloadPath:labelDownloadsAfterReload});
    await evaluate('window.beforeLabelExportReload=true;');
    await cdp("Page.reload");
    await waitFor(`window.beforeLabelExportReload!==true && typeof state!=="undefined" && state.project?.id===${JSON.stringify(labelProjectId)} && state.current && !state.busy && !element("exportLabelsButton").disabled`);
    await click("#exportLabelsButton");
    const labelsAfterReload = path.join(labelDownloadsAfterReload, "classes.txt");
    await eventually(() => fs.existsSync(labelsAfterReload));
    assert.equal(fs.readFileSync(labelsAfterReload, "utf8"), expectedLabelText, "Standalone label export still includes persisted new classes after a reload");
    assert.equal(fs.readFileSync(downloadedLabels, "utf8"), expectedLabelText, "Reuse and later exports never overwrite the original class file");
    context.diagnostic("Standalone classes.txt download, unused custom classes, Unicode, ID order, pending input, busy guards, annotation preservation, real reimport and reload passed");

    await check('!element("importLabelsButton").disabled && element("importLabelsButton").parentElement===element("exportLabelsButton").parentElement', "Label import is available beside label export in the right inspector");
    await evaluate('setBusy(true);');
    await check('element("importLabelsButton").disabled', "Label import is disabled while the workbench is busy");
    await evaluate('setBusy(false);addShape([[20,20],[80,20],[80,80],[20,80]],"rectangle");');
    const beforeLabelImport = await evaluate('JSON.stringify({current:state.current,dirty:state.dirty,selected:state.selected,activeClass:state.activeClass,undo:state.undo,redo:state.redo})');
    const beforeLabelImportSaved = await evaluate('(async()=>JSON.stringify(await api(itemUrl())))()');
    const labelImportOriginalNames = await evaluate('state.project.classes.map(label=>label.name)');
    const labelImportFile = path.join(home, "import-classes.txt");
    const labelImportText = "\ufeffImported_Raw-v3\r\n" + labelImportOriginalNames[0] + "\r\nImported_Raw-v3\r\nImported_样本\r\n";
    fs.writeFileSync(labelImportFile, labelImportText, "utf8");
    await cdp("Page.setInterceptFileChooserDialog", {enabled:true});
    await click("#importLabelsButton");
    await eventually(() => labelFileChooser);
    assert.equal(labelFileChooser.mode, "selectSingle", "The import button opens a real single-file chooser rather than asking for a path");
    await cdp("DOM.setFileInputFiles", {backendNodeId:labelFileChooser.backendNodeId, files:[labelImportFile]});
    await cdp("Page.setInterceptFileChooserDialog", {enabled:false});
    await waitFor('state.project.classes.at(-1).name==="Imported_样本" && !state.busy');
    const labelImportNames = [...labelImportOriginalNames, "Imported_Raw-v3", "Imported_样本"];
    await check('JSON.stringify(state.project.classes.map(label=>label.name))===' + JSON.stringify(JSON.stringify(labelImportNames)), "File import appends new names in file order and skips existing and duplicate names without translating");
    await check('JSON.stringify({current:state.current,dirty:state.dirty,selected:state.selected,activeClass:state.activeClass,undo:state.undo,redo:state.redo})===' + JSON.stringify(beforeLabelImport), "Import preserves unsaved boxes, selection, active class and undo history");
    await check('(async()=>JSON.stringify(await api(itemUrl()))===' + JSON.stringify(beforeLabelImportSaved) + ')()', "Import does not save or review the current image");
    const setLabelFile = async filename => {
      const document = await cdp("DOM.getDocument");
      const input = await cdp("DOM.querySelector", {nodeId:document.root.nodeId, selector:"#importLabelsFile"});
      await cdp("DOM.setFileInputFiles", {nodeId:input.nodeId, files:[filename]});
    };
    await evaluate('document.querySelectorAll(".toast").forEach(node=>node.remove());');
    await setLabelFile(labelImportFile);
    await waitFor('!state.busy && Array.from(document.querySelectorAll(".toast")).some(node=>node.textContent.includes("新增 0 个"))');
    await check('element("importLabelsFile").value==="" && state.project.classes.length===' + labelImportNames.length, "Reselecting the same file is supported and never duplicates labels");
    const invalidLabelFile = path.join(home, "invalid-labels.json");
    fs.writeFileSync(invalidLabelFile, '["Should_Not_Be_Added", 12]', "utf8");
    await setLabelFile(invalidLabelFile);
    await waitFor('!state.busy && !!document.querySelector(".toast.error")');
    await check('JSON.stringify(state.project.classes.map(label=>label.name))===' + JSON.stringify(JSON.stringify(labelImportNames)), "An invalid label file reports an error without partial imports");
    const unicodeLabelFile = path.join(home, "invalid-encoding.txt");
    fs.writeFileSync(unicodeLabelFile, Buffer.from([0xff, 0xfe, 0x41]));
    await setLabelFile(unicodeLabelFile);
    await waitFor('!state.busy && Array.from(document.querySelectorAll(".toast.error")).some(node=>node.textContent.includes("UTF-8"))');
    for (const width of [1600,1024,760]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width,height:1000,deviceScaleFactor:1,mobile:false});
      await check('(()=>{const imported=element("importLabelsButton").getBoundingClientRect(),exported=element("exportLabelsButton").getBoundingClientRect(),panel=element("importLabelsButton").closest(".inspector").getBoundingClientRect();return imported.width>0 && imported.left>=exported.right && Math.abs(imported.top-exported.top)<2 && imported.right<=panel.right;})()', "Import and export buttons remain side by side within the inspector at width " + width);
    }
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await click("#saveButton");
    await waitFor('!state.dirty && !state.busy');
    await evaluate('window.beforeLabelImportReload=true;');
    await cdp("Page.reload");
    await waitFor('window.beforeLabelImportReload!==true && typeof state!=="undefined" && state.current && !state.busy && state.project.classes.at(-1).name==="Imported_样本"');
    await check('JSON.stringify(state.project.classes.map(label=>label.name))===' + JSON.stringify(JSON.stringify(labelImportNames)), "Imported labels persist after a real browser reload");
    assert.equal(fs.readFileSync(labelImportFile, "utf8"), labelImportText, "Import never overwrites the selected source label file");
    context.diagnostic("Real file chooser, UTF-8 file upload, append-only labels, duplicate and invalid files, annotation preservation, responsive layout and reload persistence passed");

    await evaluate(`(async()=>{
      const project=await api("/api/projects","POST",{name:"Efficiency modes",image_root:${JSON.stringify(efficiencyFrames)},
        classes:["part","another"],model_path:${JSON.stringify(firstModel)}});
      state.projects.push(project);
      element("projectSelect").add(new Option(project.name,project.id));
      await selectProject(project.id);
    })()`);
    await waitFor('state.current?.item.photo_id==="part_000" && !state.busy');
    const changeMode = mode => setReviewValue("efficiencyMode", mode);
    await changeMode("fourpoint");
    await waitFor('state.tool==="fourpoint"');
    for (const xy of [[75,50],[75,117],[35,80],[115,80]]) await mouseClick(await point(xy));
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('!element("labelPicker").open && state.current.document.shapes.length===1');
    await check('state.tool==="fourpoint" && Math.abs(bounds(state.current.document.shapes[0])[0]-35)<1 && Math.abs(bounds(state.current.document.shapes[0])[3]-117)<1', "Four edge clicks yield a labeled rectangle, and the mode stays ready for the next object");
    for (const xy of [[217,150],[217,186],[198,168],[236,168]]) await mouseClick(await point(xy));
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('!element("labelPicker").open && state.current.document.shapes.length===2');
    await evaluate('selectShape(0)');
    await changeMode("fast");
    await waitFor('element("fastControls").hidden===false && state.selected===0');
    await press("]","BracketRight");
    await waitFor('state.selected===1');
    await press("2","Digit2");
    await check('(()=>{const [l,t,r,b]=bounds(state.current.document.shapes[1]),[x,y]=screenPoint([(l+r)/2,(t+b)/2]);return state.current.document.shapes[1].label==="another" && Math.abs(x-viewport().width/2)<2 && Math.abs(y-viewport().height/2)<2;})()', "Fast review centers the next box and reassigns its class from the keyboard");
    await click("#fastFull");
    await check('state.selected===-1 && state.dirty', "Returning to the whole frame preserves edits for its human review");
    await changeMode("sequence");
    await click('[data-shape="0"]');
    await click("#markFirstFrame");
    await waitFor('!state.busy && element("efficiencyHint").textContent.includes("起点 part_000")');
    await click("#inheritNext");
    await waitFor('state.current?.item.photo_id==="part_001" && !state.busy && state.current.status==="draft"');
    await check('state.current.document.shapes.length===1 && state.current.document.shapes[0].annotation_origin==="inherit_previous"', "Inheriting the selected box lands on the next same-sequence frame with a draft");

    const endpointFrame = await evaluate('state.items.find(item=>item.photo_id==="part_003").id');
    await click(`[data-item="${endpointFrame}"]`);
    await waitFor('state.current?.item.photo_id==="part_003" && !state.busy');
    await changeMode("prompt");
    await waitFor('state.tool==="prompt"');
    await mouseClick(await point([125,78]));
    await waitFor('!element("acceptPrompt").disabled');
    await click("#promptNegative");
    await mouseClick(await point([258,166]));
    await waitFor('!element("acceptPrompt").disabled');
    await click("#acceptPrompt");
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('!element("labelPicker").open && state.current.document.shapes.length===1');
    await check('bounds(state.current.document.shapes[0]).every((coord,index)=>Math.abs(coord-[101,50,181,117][index])<1)', "Real local point prompting proposes the target box and accepts it through the standard label picker");
    await changeMode("sequence");
    await click('[data-shape="0"]');
    await click("#previewLastFrame");
    await waitFor('element("sequenceDialog").open && element("sequenceTarget").options[0]?.text==="part_002"');
    await check('element("sequenceSummary").textContent.includes("1 张草稿") && element("sequencePreview").querySelector(".comparison-frame-box")!==null', "Keyframe interpolation previews the intermediate box and skips the already inherited draft");
    await click("#sequenceApply");
    await waitFor('!element("sequenceDialog").open && state.items.find(item=>item.photo_id==="part_002")?.status==="draft"');

    await click("#similarityButton");
    await click("#scanSimilarity");
    await waitFor('!element("similarityResults").hidden && element("similarityCards").querySelectorAll("input").length===2');
    await click("#similarityKeepRecommended");
    await check('element("similaritySelectionInfo").textContent.includes("队列保留 1 张") && element("similarityCards").querySelectorAll("input:checked").length===1', "Visual screening groups a PNG and its re-encoded JPEG and allows retaining the representative");
    const similarityScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"similarity-review.png"), Buffer.from(similarityScreenshot.data,"base64"));
    await click("#applySimilarity");
    await waitFor('!element("similarityDialog").open && !state.busy && element("statusSelect").value==="curated" && state.items.length===1');

    await click("#auditButton");
    await click("#startAudit");
    await waitFor('element("auditStatus").textContent.includes("扫描完成") && element("auditStatus").textContent.includes("疑点 1 张")');
    await click("#openAuditQueue");
    await waitFor('!element("auditFrameBar").hidden && state.current?.item.photo_id==="part_002"');
    await check('element("auditFrameReasons").textContent.includes("类别冲突 1") && state.current.document.shapes[0].label==="part"', "Suspicion review opens the flagged frame next to its unchanged manual label");
    await click("#toggleAuditOverlay");
    await check('element("toggleAuditOverlay").getAttribute("aria-pressed")==="false"', "The audit hint overlay can be toggled while the frame stays editable");
    await click("#dismissAuditFrame");
    await waitFor('state.items.length===0 && element("auditFrameBar").hidden && !state.current');

    await setReviewValue("statusSelect", "");
    await waitFor('!state.busy && state.items.length===6 && state.current?.item.photo_id==="part_000"');
    const resumeItem = await evaluate('state.items.find(item=>item.photo_id==="part_003").id');
    const prefetchItem = await evaluate('state.items.find(item=>item.photo_id==="part_004").id');
    await click(`[data-item="${resumeItem}"]`);
    await waitFor('state.current?.item.photo_id==="part_003" && !state.busy');
    await waitFor(`performance.getEntriesByType("resource").some(entry=>entry.name.endsWith("/items/${prefetchItem}/image"))`);
    await check('!element("autoDraft").checked', "The draft preference survives workspace navigation");
    await click("#autoDraft");
    await box([21,158],[55,185]);
    await press("q","KeyQ");
    await waitFor('state.current.document.shapes.length===2 && !state.dirty && element("autoSaveStatus").textContent.includes("自动保存")');
    const savedVersion = await evaluate('state.current.version');
    await evaluate('zoom(1.35); window.beforeResumeReload=true;');
    const savedZoom = await evaluate('state.scale');
    await sleep(280);
    await cdp("Page.reload");
    await waitFor(`window.beforeResumeReload!==true && typeof state!=="undefined" && !state.busy && state.current?.item.id===${JSON.stringify(resumeItem)}`);
    await check(`state.current.version===${savedVersion} && state.current.status==="draft" && state.selected===1 && Math.abs(state.scale-${savedZoom})<.001`, "Idle editing saves a draft and a reload restores the image, box selection and zoom");

    await click("#automationButton");
    await waitFor('element("automationDialog").open && !element("runIntake").disabled');
    const originalWorkflowFolder = await evaluate('element("workflowFolder").value');
    await click("#browseWorkflowFolder");
    await waitFor('element("imageBrowserDialog").open && imageBrowser.valid && !imageBrowser.loading');
    await check('imageBrowser.target==="workflowFolder" && imageBrowser.mode==="folder"', "Automation retains its local folder browser alongside native project image selection");
    await click("#cancelImageBrowser");
    await check(`element("workflowFolder").value===${JSON.stringify(originalWorkflowFolder)} && element("automationDialog").open`, "Cancelling the automation folder browser preserves its recipe and open dialog");
    await click("#workflowRepresentatives");
    await click("#runIntake");
    await waitFor('document.querySelector("[data-workflow-run]")?.textContent.includes("待审核 1 张")');
    await check('element("workflowRuns").textContent.includes("暂缓 1 张") && !!element("workflowRuns").querySelector("[data-workflow-action=triage]")', "The new batch pipeline groups similar images, prelabels the representative and exposes its review queue");
    const workflowScreenshot = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(home,"automation.png"), Buffer.from(workflowScreenshot.data,"base64"));
    await click('[data-workflow-action="triage"]');
    await waitFor('!element("automationDialog").open && element("statusSelect").value==="triage" && state.items.length===1 && !state.busy');
    await check('state.current.status==="predicted" && element("imageList").textContent.includes("无候选框")', "Automated routing keeps empty outputs pending for human review");

    const createTask = async (task, label) => {
      await click("#newProjectButton");
      await waitFor('element("projectDialog").open');
      await click(`.task-option:has(input[value="${task}"])`);
      await setExportValue("newProjectName", task);
      await setExportValue("newClasses", label);
      await evaluate(`element("newImageRoot").value=${JSON.stringify(efficiencyFrames)}; element("newImageRoot").dispatchEvent(new Event("input"));`);
      await click("#createProject");
      await waitFor(`state.project.task_type==="${task}" && state.current && !state.busy && !element("projectDialog").open`);
    };
    if (await evaluate('element("autoDraft").checked')) await click("#autoDraft");
    await click("#newProjectButton");
    await check('document.querySelector(".logo img").naturalWidth>0 && document.querySelectorAll(".task-option").length===4',
      "The RealityLoop logo loads and the project dialog offers four task types");
    const taskNamesFit = `Array.from(document.querySelectorAll(".task-option")).every(tile =>
      tile.querySelector("span").getBoundingClientRect().right <= tile.getBoundingClientRect().right)`;
    await check(taskNamesFit, "Task names and descriptions fit inside the desktop choices");
    fs.writeFileSync(path.join(home,"task-selection.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data,"base64"));
    await cdp("Emulation.setDeviceMetricsOverride", {width:420,height:850,deviceScaleFactor:1,mobile:false});
    await sleep(80);
    await check(taskNamesFit, "Task names and descriptions also fit on a narrow screen");
    await cdp("Emulation.setDeviceMetricsOverride", {width:1600,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('element("projectDialog").close()');
    await createTask("segment", "part");
    await check(`element("projectTask").textContent==="实例分割" && document.querySelector('[data-tool="rectangle"]').hidden`,
      "Instance projects show contour tools and a task-specific header");
    await click('[data-tool="polygon"]');
    for (const coords of [[35,50],[114,50],[114,116],[35,116]]) await mouseClick(await point(coords));
    await press("q","KeyQ");
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('state.current.document.shapes[0]?.shape_type==="polygon"');
    await changeMode("prompt");
    await mouseClick(await point([60,73]));
    await waitFor('!element("acceptPrompt").disabled');
    await click("#acceptPrompt");
    await waitFor('element("labelPicker").open');
    await press("q","KeyQ");
    await waitFor('state.current.document.shapes.length===2');
    await check('state.current.document.shapes.every(shape=>shape.shape_type==="polygon")',
      "Manual contours and point-assisted proposals both create independent instance polygons");
    await click("#saveButton");
    await waitFor('!state.busy && !state.dirty');

    await createTask("semantic", "region");
    await click('[data-tool="brush"]');
    await mouseClick(await point([61,80]));
    await click('[data-tool="erase"]');
    await setExportValue("brushRadius", "6");
    await mouseClick(await point([61,80]));
    await check('state.current.document.shapes.length===2 && state.current.document.shapes[1].erase===true && state.current.document.shapes[1].radius===6',
      "Semantic painting and erasing retain ordered pixel regions");
    await press("e","KeyE");
    await waitFor('state.items.some(item=>item.status==="reviewed") && !state.busy');
    await click("#exportButton");
    await waitFor('element("exportDialog").open');
    if (await evaluate('element("splitDataset").checked')) await click("#splitDataset");
    await waitFor('element("exportDialog").open && exportPreview.plan');
    await check('element("exportFormat").value==="mask" && !exportPreview.plan.data.blocking_issue',
      "Semantic export is ready with a per-pixel PNG mask");
    await evaluate('element("exportDialog").close()');

    await createTask("hand_pose", "");
    await click('[data-tool="hand"]');
    await mouseClick(await point([130,147]));
    await click("#skipHandPoint");
    await mouseClick(await point([105,113]));
    await click("#finishHand");
    await waitFor('state.current.document.shapes[0]?.keypoints.length===21');
    await check('state.project.classes[0].name==="hand" && state.current.document.shapes[0].keypoints[1][2]===0',
      "Hand annotation stores the 21-point template with skipped keypoints");
    await setReviewValue("jointSelect", "2");
    await setReviewValue("jointVisibility", "1");
    await click("#placeJoint");
    await mouseClick(await point([112,105]));
    await check('state.current.document.shapes[0].keypoints[2][2]===1 && Math.abs(state.current.document.shapes[0].keypoints[2][0]-112)<2',
      "Hand keypoints can be relocated and marked occluded");
    await click("#saveButton");
    await waitFor('state.current.status==="draft" && !state.dirty && !state.busy');
    fs.writeFileSync(path.join(home,"hand-workspace.png"), Buffer.from((await cdp("Page.captureScreenshot", {format:"png"})).data,"base64"));

    assert.deepEqual(errors, [], "No browser runtime errors");
  } finally {
    if (socket?.readyState === WebSocket.OPEN && cdp) await cdp("Browser.close").catch(() => {});
    socket?.close();
    browser?.kill();
    server.kill();
    context.diagnostic(`Isolated fixtures: ${home}`);
  }
});
