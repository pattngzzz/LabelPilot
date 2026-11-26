const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const root = path.resolve(__dirname, "..");
const app = fs.readFileSync(path.join(root, "static/app.js"), "utf8");
const tools = fs.readFileSync(path.join(root, "static/task-tools.js"), "utf8");

function section(source, start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first);
  return source.slice(first, last);
}

function harness(task="ocr", shape={label:"text", shape_type:"rectangle", points:[[40,40],[160,80]],
    text_raw:"00.250 g", readability:"readable", unit:"g", value_role:"process_value",
    value_decimal:"00.250", flags:{verified:true}, attributes:{source:"manual"}}) {
  const state = {current:{width:240,height:180,document:{shapes:[structuredClone(shape)]}},
    selected:0, undo:[], redo:[], tool:"select", busy:false, drag:null, pendingShape:null,
    rectangle:null, polygon:[], space:false, dirty:false};
  const environment = {state, dialog:false, notifications:[], renders:0};
  let onKeydown;
  const sandbox = {
    state, copy:structuredClone, stroke:null, newHand:null, placing:null,
    document:{
      addEventListener:(name, callback) => { assert.equal(name, "keydown"); onKeydown = callback; },
      querySelector:selector => {
        if (selector === "dialog[open]") return environment.dialog ? {} : null;
        const tool = /\[data-tool="(\w+)"\]/.exec(selector)?.[1];
        return tool ? {hidden:tool === "hand" ? task !== "hand_pose" : task !== "semantic"} : null;
      },
    },
    toast:(message, error) => environment.notifications.push({message,error}),
    setDirty:() => { state.dirty = true; },
    renderObjects:() => { environment.renders++; }, draw:() => {},
    setTool:tool => { state.tool = tool; },
    efficiency:{keydown:() => false},
  };
  vm.createContext(sandbox);
  vm.runInContext(section(app, "function snapshot()", "function classInfo("), sandbox);
  vm.runInContext(section(app, "function bounds(", "function pathPoints("), sandbox);
  vm.runInContext(section(tools, "  function toolAllowed(", "  function toolChanged("), sandbox);
  vm.runInContext(section(tools, "  function keydown(", "  function opacity("), sandbox);
  sandbox.taskUi = {type:() => task, keydown:sandbox.keydown};
  vm.runInContext(section(app, 'document.addEventListener("keydown",event=>{', "\ntoggleFocusMode("), sandbox);
  environment.press = (key, options={}) => {
    const event = {key, code:`Key${key.toUpperCase()}`, target:{tagName:"CANVAS"},
      preventDefault:() => { event.prevented = true; }, ...options};
    onKeydown(event);
    return event;
  };
  return environment;
}

function close(actual, expected) {
  assert.ok(Math.abs(actual-expected) < 1e-8, `${actual} differs from ${expected}`);
}

function pointsClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((point, index) => point.forEach((coordinate, axis) => close(coordinate, expected[index][axis])));
}

function angle(shape) {
  return Math.atan2(shape.points[1][1]-shape.points[0][1], shape.points[1][0]-shape.points[0][0])*180/Math.PI;
}

for (const task of ["detect", "segment", "ocr"]) {
  test(`${task}: Z and X rotate the selected box counterclockwise and clockwise`, () => {
    for (const [key, degrees] of [["z",-1],["x",1]]) {
      const environment = harness(task);
      const original = structuredClone(environment.state.current.document.shapes[0]);
      assert.equal(environment.press(key).prevented, true);
      const rotated = environment.state.current.document.shapes[0];
      assert.equal(rotated.shape_type, "polygon");
      assert.equal(rotated.points.length, 4);
      close(angle(rotated), degrees);
      close(Math.hypot(rotated.points[1][0]-rotated.points[0][0], rotated.points[1][1]-rotated.points[0][1]), 120);
      close(Math.hypot(rotated.points[2][0]-rotated.points[1][0], rotated.points[2][1]-rotated.points[1][1]), 40);
      close(rotated.points.reduce((sum, point) => sum+point[0]/4, 0), 100);
      close(rotated.points.reduce((sum, point) => sum+point[1]/4, 0), 60);
      close(rotated.rotation, degrees);
      const {points, shape_type, rotation, ...attributes} = rotated;
      const {points:originalPoints, shape_type:originalType, ...originalAttributes} = original;
      assert.deepEqual(attributes, originalAttributes);
      assert.equal(environment.state.dirty, true);
      assert.equal(environment.state.undo.length, 1);
      assert.equal(environment.state.selected, 0);
    }
  });
}

test("Shift rotates 5 degrees, repeated keys continue rotating without changing size or center", () => {
  const environment = harness();
  environment.press("X", {shiftKey:true});
  close(angle(environment.state.current.document.shapes[0]), 5);
  for (let index=1; index<72; index++) environment.press("X", {shiftKey:true, repeat:true});
  const points = environment.state.current.document.shapes[0].points;
  pointsClose(points, [[40,40],[160,40],[160,80],[40,80]]);
  environment.press("Z", {shiftKey:true});
  close(angle(environment.state.current.document.shapes[0]), -5);
});

test("Rotation works on selected four-point polygons and leaves other objects untouched", () => {
  const environment = harness("ocr", {label:"text", shape_type:"polygon", points:[[42,30],[155,42],[160,80],[38,70]]});
  const original = structuredClone(environment.state.current.document.shapes[0]);
  environment.state.current.document.shapes.push(structuredClone(original));
  environment.state.selected = 1;
  environment.press("x");
  assert.deepEqual(environment.state.current.document.shapes[0], original);
  environment.press("z");
  pointsClose(environment.state.current.document.shapes[1].points, original.points);
});

test("Ctrl+Z and Ctrl+Shift+Z still undo and redo the full rotation including shape type", () => {
  const environment = harness();
  const original = structuredClone(environment.state.current.document);
  environment.press("x");
  const rotated = structuredClone(environment.state.current.document);
  environment.press("z", {ctrlKey:true});
  assert.deepEqual(environment.state.current.document, original);
  environment.press("Z", {ctrlKey:true, shiftKey:true});
  assert.deepEqual(environment.state.current.document, rotated);
  environment.press("z", {metaKey:true});
  assert.deepEqual(environment.state.current.document, original);
});

test("Rotations can cross image boundaries without clipping, resizing or moving the box", () => {
  const environment = harness("ocr", {label:"text", shape_type:"rectangle", points:[[0,0],[240,180]]});
  const original = structuredClone(environment.state.current.document);
  environment.state.redo.push(structuredClone(original));
  environment.press("x");
  environment.press("x", {repeat:true});
  const shape = environment.state.current.document.shapes[0];
  close(angle(shape), 2);
  assert.ok(shape.points.some(([horizontal,vertical]) => horizontal<0 || vertical<0 || horizontal>240 || vertical>180));
  close(Math.hypot(shape.points[1][0]-shape.points[0][0], shape.points[1][1]-shape.points[0][1]), 240);
  close(Math.hypot(shape.points[2][0]-shape.points[1][0], shape.points[2][1]-shape.points[1][1]), 180);
  close(shape.points.reduce((sum, point) => sum+point[0]/4, 0), 120);
  close(shape.points.reduce((sum, point) => sum+point[1]/4, 0), 90);
  assert.equal(environment.state.undo.length, 2);
  assert.equal(environment.state.redo.length, 0);
  assert.equal(environment.state.dirty, true);
  assert.equal(environment.notifications.length, 0);
});

test("No rotation during drawing, dragging, saving, panning or when nothing is selected", () => {
  for (const blocked of [{rectangle:{start:[1,1]}}, {polygon:[[1,1]]}, {pendingShape:{}},
    {drag:{type:"move"}}, {busy:true}, {space:true}, {selected:-1}, {current:null}]) {
    const environment = harness();
    Object.assign(environment.state, blocked);
    const original = structuredClone(environment.state.current);
    environment.press("x");
    assert.deepEqual(environment.state.current, original);
    assert.equal(environment.state.undo.length, 0);
    assert.equal(environment.state.dirty, false);
  }
});

test("Text input, IME, dialogs and modified shortcuts never rotate an annotation", () => {
  for (const options of [{target:{tagName:"INPUT"}}, {target:{tagName:"TEXTAREA"}},
    {target:{tagName:"SELECT"}}, {target:{tagName:"DIV",isContentEditable:true}},
    {isComposing:true}, {ctrlKey:true}, {metaKey:true}, {altKey:true}, {dialog:true}]) {
    const environment = harness();
    environment.dialog = Boolean(options.dialog);
    const original = structuredClone(environment.state.current.document);
    environment.press("x", options);
    assert.deepEqual(environment.state.current.document, original);
    assert.equal(environment.state.undo.length, 0);
  }
});

test("Semantic X remains the eraser and rotation cannot alter masks, hands or general contours", () => {
  for (const task of ["semantic", "hand_pose"]) {
    const environment = harness(task);
    const original = structuredClone(environment.state.current.document);
    environment.press("x");
    environment.press("z");
    assert.deepEqual(environment.state.current.document, original);
    assert.equal(environment.state.tool, task === "semantic" ? "erase" : "select");
    assert.equal(environment.state.undo.length, 0);
  }
  for (const shape of [{shape_type:"polygon", points:[[1,1],[50,1],[50,40]]},
    {shape_type:"polygon", points:[[1,1],[50,1],[60,20],[50,40],[1,40]]},
    {shape_type:"mask", points:[[1,1],[50,40]]}, {shape_type:"hand", points:[[1,1],[50,40]]}]) {
    const environment = harness("detect", shape);
    const original = structuredClone(environment.state.current.document);
    environment.press("x");
    assert.deepEqual(environment.state.current.document, original);
    assert.equal(environment.state.undo.length, 0);
  }
});
