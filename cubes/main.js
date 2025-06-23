
const canvas = document.getElementById('gfx');
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
const stats = document.getElementById('stats');

// lower device pixel ratio to lighten GPU load
const dpr = 1;
canvas.width = canvas.clientWidth * dpr;
canvas.height = canvas.clientHeight * dpr;

const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format,
  alphaMode: 'opaque'
});

// Camera control - orbit camera around the origin
let radius = 170;
let yaw = Math.PI / 4;
const cameraPos = [radius * Math.sin(yaw), 100, radius * Math.cos(yaw)];
const keyState = {};
const cameraSpeed = 50;
let lastTime = 0;
let accum = 0;
let frames = 0;
let lastFpsUpdate = 0;

window.addEventListener('keydown', (e) => { keyState[e.key.toLowerCase()] = true; });
window.addEventListener('keyup', (e) => { keyState[e.key.toLowerCase()] = false; });

function updateCamera(dt) {
  const step = cameraSpeed * dt;
  if (keyState['w']) radius = Math.max(10, radius - step);
  if (keyState['s']) radius += step;
  if (keyState['a']) yaw += step * 0.01;
  if (keyState['d']) yaw -= step * 0.01;
  cameraPos[0] = radius * Math.sin(yaw);
  cameraPos[2] = radius * Math.cos(yaw);
}

const shaderCode = await (await fetch('shaders.wgsl')).text();
const shaderModule = device.createShaderModule({ code: shaderCode });

const cubeVerts = new Float32Array([
  -0.5, -0.5, -0.5,
   0.5, -0.5, -0.5,
   0.5,  0.5, -0.5,
  -0.5,  0.5, -0.5,
  -0.5, -0.5,  0.5,
   0.5, -0.5,  0.5,
   0.5,  0.5,  0.5,
  -0.5,  0.5,  0.5,
]);

const cubeIndices = new Uint16Array([
  0,1,2, 2,3,0, 4,5,6, 6,7,4,
  0,1,5, 5,4,0, 2,3,7, 7,6,2,
  0,3,7, 7,4,0, 1,2,6, 6,5,1
]);

function createBuffer(arr, usage) {
  const buffer = device.createBuffer({
    size: arr.byteLength,
    usage,
    mappedAtCreation: true,
  });
  const view = usage & GPUBufferUsage.INDEX
    ? new Uint16Array(buffer.getMappedRange())
    : new Float32Array(buffer.getMappedRange());
  view.set(arr);
  buffer.unmap();
  return buffer;
}

const vertexBuffer = createBuffer(cubeVerts, GPUBufferUsage.VERTEX);
const indexBuffer = createBuffer(cubeIndices, GPUBufferUsage.INDEX);

const gridSize = 20;
const spacing = 2;
const numInstances = gridSize ** 3;

const positions = new Float32Array(numInstances * 3);
const colors = new Float32Array(numInstances * 3);

let i = 0;
for (let x = 0; x < gridSize; x++) {
  for (let y = 0; y < gridSize; y++) {
    for (let z = 0; z < gridSize; z++) {
      positions[i * 3 + 0] = (x - gridSize / 2) * spacing;
      positions[i * 3 + 1] = (y - gridSize / 2) * spacing;
      positions[i * 3 + 2] = (z - gridSize / 2) * spacing;
      colors[i * 3 + 0] = 1.0;
      colors[i * 3 + 1] = 1.0;
      colors[i * 3 + 2] = 1.0;
      i++;
    }
  }
}

const instanceBuffer = createBuffer(positions, GPUBufferUsage.VERTEX);
const colorBuffer = createBuffer(colors, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

const vpBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: shaderModule,
    entryPoint: 'vs_main',
    buffers: [
      {
        arrayStride: 12,
        attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }]
      },
      {
        arrayStride: 12,
        stepMode: 'instance',
        attributes: [{ shaderLocation: 1, format: 'float32x3', offset: 0 }]
      }
    ]
  },
  fragment: {
    module: shaderModule,
    entryPoint: 'fs_main',
    targets: [{ format }]
  },
  primitive: { topology: 'triangle-list', cullMode: 'back' },
  depthStencil: {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less'
  }
});

const vpBindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vpBuffer } },
    { binding: 1, resource: { buffer: colorBuffer } }
  ]
});

const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: 'depth24plus',
  usage: GPUTextureUsage.RENDER_ATTACHMENT
});

const vpMatrix = new Float32Array(16);
const perspective = new Float32Array(16);
const view = new Float32Array(16);

function mat4Perspective(out, fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
  out[8] = 0; out[9] = 0; out[10] = (far) / (near - far); out[11] = -1;
  out[12] = 0; out[13] = 0; out[14] = (near * far) / (near - far); out[15] = 0;
}

function mat4LookAt(out, eye, center, up) {
  let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;

  z0 = eye[0] - center[0];
  z1 = eye[1] - center[1];
  z2 = eye[2] - center[2];
  len = Math.hypot(z0, z1, z2);
  z0 /= len; z1 /= len; z2 /= len;

  x0 = up[1] * z2 - up[2] * z1;
  x1 = up[2] * z0 - up[0] * z2;
  x2 = up[0] * z1 - up[1] * z0;
  len = Math.hypot(x0, x1, x2);
  x0 /= len; x1 /= len; x2 /= len;

  y0 = z1 * x2 - z2 * x1;
  y1 = z2 * x0 - z0 * x2;
  y2 = z0 * x1 - z1 * x0;

  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1;
}

function updateVPMatrix() {
  const eye = cameraPos;
  const center = [0, 0, 0];
  const up = [0, 1, 0];
  mat4Perspective(perspective, Math.PI / 4, canvas.width / canvas.height, 0.1, 2000);
  mat4LookAt(view, eye, center, up);

  for (let c = 0; c < 4; ++c) {
    for (let r = 0; r < 4; ++r) {
      vpMatrix[c * 4 + r] =
        perspective[0 * 4 + r] * view[c * 4 + 0] +
        perspective[1 * 4 + r] * view[c * 4 + 1] +
        perspective[2 * 4 + r] * view[c * 4 + 2] +
        perspective[3 * 4 + r] * view[c * 4 + 3];
    }
  }

  device.queue.writeBuffer(vpBuffer, 0, vpMatrix);
}

function frame(timeMs) {
  const dt = (timeMs - lastTime) / 1000;
  lastTime = timeMs;
  updateCamera(dt);
  accum += dt;
  if (accum < 1 / 30) {
    requestAnimationFrame(frame);
    return;
  }
  accum = 0;
  frames++;
  updateVPMatrix();

  const cmd = device.createCommandEncoder();

  const render = cmd.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store'
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store'
    }
  });

  render.setPipeline(pipeline);
  render.setBindGroup(0, vpBindGroup);
  render.setVertexBuffer(0, vertexBuffer);
  render.setVertexBuffer(1, instanceBuffer);
  render.setIndexBuffer(indexBuffer, 'uint16');
  render.drawIndexed(cubeIndices.length, numInstances);
  render.end();

  device.queue.submit([cmd.finish()]);

  if (timeMs - lastFpsUpdate > 500) {
    const fps = (frames * 1000) / (timeMs - lastFpsUpdate);
    stats.textContent = `FPS: ${fps.toFixed(1)} | Rendered: ${numInstances} | Total: ${numInstances}`;
    frames = 0;
    lastFpsUpdate = timeMs;
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

