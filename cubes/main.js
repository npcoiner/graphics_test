// WebGPU cube grid demo with animated colors and extended grid range
// Controls   : panel bottom‑left (grid‑size slider)
// FPS counter: top‑left
// WASD keys to move camera
// --------------------------------------------------
const canvas = document.getElementById('gfx');
const keyState = {};
window.addEventListener('keydown', e => keyState[e.key.toLowerCase()] = true);
window.addEventListener('keyup', e => keyState[e.key.toLowerCase()] = false);

/* HUD (FPS) */
const stats = (() => {
  let d = document.getElementById('stats');
  if (!d) {
    d = document.createElement('div');
    Object.assign(d.style, {
      position: 'fixed', top: '6px', left: '6px', zIndex: 1e4,
      background: 'rgba(0,0,0,.6)', color: '#fff', fontFamily: 'monospace', padding: '4px 8px'
    });
    document.body.appendChild(d);
  }
  return d;
})();

/* Small control panel */
const panel = document.createElement('div');
Object.assign(panel.style, {
  position: 'fixed', bottom: '10px', left: '10px', background: 'rgba(0,0,0,.6)',
  color: '#fff', padding: '10px', fontFamily: 'monospace', zIndex: 1e4
});
document.body.appendChild(panel);

const adapter = await navigator.gpu.requestAdapter();
const adapterLimits = adapter.limits;
const safeLimit = Math.floor(adapterLimits.maxBufferSize * 0.25);
const maxFloats = Math.floor(safeLimit / 4);
const maxVec3Count = Math.floor(maxFloats / 3);
const maxGridSize = Math.floor(Math.cbrt(maxVec3Count));

const slider = Object.assign(document.createElement('input'), { type: 'range', min: 1, max: maxGridSize, value: 6 });
const sizeLabel = document.createElement('span'); sizeLabel.textContent = slider.value;
panel.append('Grid: ', slider, sizeLabel);

let gridSize = +slider.value;
let spacing  = 2;
let numInstances = 0;
let instanceBuffer;

const device  = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapterLimits.maxBufferSize
  }
});
const context = canvas.getContext('webgpu');
const format  = navigator.gpu.getPreferredCanvasFormat();
const dpr     = window.devicePixelRatio || 1;
let depthTex;
function resize() {
  canvas.width  = canvas.clientWidth  * dpr;
  canvas.height = canvas.clientHeight * dpr;
  context.configure({ device, format, alphaMode: 'opaque' });
  depthTex = device.createTexture({ size: [canvas.width, canvas.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
}
resize();
window.addEventListener('resize', resize);

function makeBuffer(data, usage) {
  const b = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
  (usage & GPUBufferUsage.INDEX ? new Uint16Array(b.getMappedRange()) : new Float32Array(b.getMappedRange())).set(data);
  b.unmap(); return b;
}

const cubeVerts = new Float32Array([
  -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5, 0.5,-0.5, -0.5, 0.5,-0.5,
  -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5
]);
const cubeIdx = new Uint16Array([
  0,1,2, 2,3,0, 4,5,6, 6,7,4,
  0,1,5, 5,4,0, 2,3,7, 7,6,2,
  0,3,7, 7,4,0, 1,2,6, 6,5,1
]);
const vbuf = makeBuffer(cubeVerts, GPUBufferUsage.VERTEX);
const ibuf = makeBuffer(cubeIdx , GPUBufferUsage.INDEX);

function rebuildInstances() {
  numInstances = gridSize ** 3;
  const arr = new Float32Array(numInstances * 3);
  let k = 0;
  for (let x = 0; x < gridSize; ++x)
    for (let y = 0; y < gridSize; ++y)
      for (let z = 0; z < gridSize; ++z) {
        arr[k++] = (x - gridSize / 2) * spacing;
        arr[k++] = (y - gridSize / 2) * spacing;
        arr[k++] = (z - gridSize / 2) * spacing;
      }
  instanceBuffer = makeBuffer(arr, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
}
rebuildInstances();

const wgsl = `
struct VP { m : mat4x4<f32> };
struct Time { t : f32 };
@group(0) @binding(0) var<uniform> vp : VP;
@group(0) @binding(1) var<uniform> time : Time;
struct In { @location(0) p : vec3<f32>, @location(1) inst : vec3<f32>, @builtin(instance_index) i : u32 };
struct Out { @builtin(position) pos : vec4<f32>, @location(0) col : vec3<f32> };
@vertex fn vs_main(i:In)->Out{
  var o:Out;
  let w=i.p+i.inst;
  o.pos=vp.m*vec4<f32>(w,1);
  let t = time.t + f32(i.i) * 0.1;
  o.col = vec3<f32>(0.5+0.5*sin(t), 0.5+0.5*sin(t+2.1), 0.5+0.5*sin(t+4.2));
  return o;
}
@fragment fn fs_main(i:Out)->@location(0) vec4<f32>{ return vec4<f32>(i.col,1); }`;
const shaderModule = device.createShaderModule({ code: wgsl });

let pipeline, bindGroup;
const vpBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const timeBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
function buildPipeline() {
  pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule, entryPoint: 'vs_main',
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] },
        { arrayStride: 12, stepMode: 'instance', attributes: [{ shaderLocation: 1, format: 'float32x3', offset: 0 }] }
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  });
  bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: vpBuf } },
    { binding: 1, resource: { buffer: timeBuf } }
  ] });
}
buildPipeline();

slider.oninput = () => { gridSize = +slider.value; sizeLabel.textContent = slider.value; rebuildInstances(); };

function persp(out, fovy, asp, n, f) {
  const t = Math.tan(fovy / 2);
  out.fill(0); out[0] = 1 / (asp * t); out[5] = 1 / t; out[10] = f / (n - f); out[11] = -1; out[14] = (n * f) / (n - f);
}
function lookAt(out, e, c, u) {
  const zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2];
  const zl = 1 / Math.hypot(zx, zy, zz);
  const nx = zx * zl, ny = zy * zl, nz = zz * zl;
  const xx = u[1] * nz - u[2] * ny, xy = u[2] * nx - u[0] * nz, xz = u[0] * ny - u[1] * nx;
  const xl = 1 / Math.hypot(xx, xy, xz);
  const x0 = xx * xl, x1 = xy * xl, x2 = xz * xl;
  const y0 = ny * x2 - nz * x1, y1 = nz * x0 - nx * x2, y2 = nx * x1 - ny * x0;
  out[0] = x0; out[1] = y0; out[2] = nx; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = ny; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = nz; out[11] = 0;
  out[12] = -(x0 * e[0] + x1 * e[1] + x2 * e[2]);
  out[13] = -(y0 * e[0] + y1 * e[1] + y2 * e[2]);
  out[14] = -(nx * e[0] + ny * e[1] + nz * e[2]);
  out[15] = 1;
}

const vpMat = new Float32Array(16), perspMat = new Float32Array(16), viewMat = new Float32Array(16);
let lastTime = 0, frameCount = 0, fpsUpdate = 0;
let radius = 320, angle = Math.PI / 4;

function frame(time) {
  if (!lastTime) lastTime = time;
  const dt = (time - lastTime) / 1000;
  lastTime = time;
  frameCount++;
  if (time - fpsUpdate > 1000) {
    stats.textContent = `FPS: ${(frameCount * 1000 / (time - fpsUpdate)).toFixed(1)} | Cubes: ${numInstances}`;
    fpsUpdate = time;
    frameCount = 0;
  }

  if (keyState['w']) radius -= 100 * dt;
  if (keyState['s']) radius += 100 * dt;
  if (keyState['a']) angle += 1.5 * dt;
  if (keyState['d']) angle -= 1.5 * dt;

  const eyeX = radius * Math.cos(angle);
  const eyeZ = radius * Math.sin(angle);
  const eye = [eyeX, radius, eyeZ];
  persp(perspMat, Math.PI/4, canvas.width/canvas.height, 0.1, 2000);
  lookAt(viewMat, eye, [0,0,0], [0,1,0]);
  for (let c = 0; c < 4; ++c)
    for (let r = 0; r < 4; ++r)
      vpMat[c * 4 + r] =
        perspMat[0 * 4 + r] * viewMat[c * 4 + 0] +
        perspMat[1 * 4 + r] * viewMat[c * 4 + 1] +
        perspMat[2 * 4 + r] * viewMat[c * 4 + 2] +
        perspMat[3 * 4 + r] * viewMat[c * 4 + 3];

  const now = performance.now() / 1000;
  device.queue.writeBuffer(timeBuf, 0, new Float32Array([now]));
  device.queue.writeBuffer(vpBuf, 0, vpMat);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 }
    }],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1,
      depthStoreOp: 'store'
    }
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vbuf);
  pass.setVertexBuffer(1, instanceBuffer);
  pass.setIndexBuffer(ibuf, 'uint16');
  pass.drawIndexed(cubeIdx.length, numInstances);
  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

