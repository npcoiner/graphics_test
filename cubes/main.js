const canvas = document.getElementById('gfx');
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  alert('WebGPU not supported');
  throw new Error('WebGPU not supported');
}
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');

const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format: presentationFormat,
  alphaMode: 'opaque'
});

// Load shader code
const response = await fetch('shaders.wgsl');
const shaderCode = await response.text();
const shaderModule = device.createShaderModule({ code: shaderCode });

// Cube geometry data
const cubeVertices = new Float32Array([
  // position
  -1,-1,-1,
   1,-1,-1,
   1, 1,-1,
  -1, 1,-1,
  -1,-1, 1,
   1,-1, 1,
   1, 1, 1,
  -1, 1, 1,
]);

const cubeIndices = new Uint16Array([
  0,1,2, 2,3,0,
  4,5,6, 6,7,4,
  0,1,5, 5,4,0,
  2,3,7, 7,6,2,
  0,3,7, 7,4,0,
  1,2,6, 6,5,1,
]);

// Create GPU buffers
function createBuffer(arr, usage) {
  const buffer = device.createBuffer({
    size: arr.byteLength,
    usage,
    mappedAtCreation: true
  });
  const mapping = usage & GPUBufferUsage.INDEX ? new Uint16Array(buffer.getMappedRange()) : new Float32Array(buffer.getMappedRange());
  mapping.set(arr);
  buffer.unmap();
  return buffer;
}

const vertexBuffer = createBuffer(cubeVertices, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
const indexBuffer = createBuffer(cubeIndices, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);

const numInstances = 1000;
// Instance positions
const instanceStride = 3;
const instancePositions = new Float32Array(numInstances * instanceStride);
for (let i = 0; i < numInstances; ++i) {
  instancePositions[i * 3 + 0] = (Math.random() - 0.5) * 50;
  instancePositions[i * 3 + 1] = (Math.random() - 0.5) * 50;
  instancePositions[i * 3 + 2] = (Math.random() - 0.5) * 50;
}
const instanceBuffer = createBuffer(instancePositions, GPUBufferUsage.VERTEX);

// Color + offset buffer
const colorStride = 4; // vec4
const colors = new Float32Array(numInstances * colorStride);
for (let i = 0; i < numInstances; ++i) {
  const offset = Math.random() * 1000.0;
  colors[i * 4 + 0] = 1.0;
  colors[i * 4 + 1] = 1.0;
  colors[i * 4 + 2] = 1.0;
  colors[i * 4 + 3] = offset;
}
const colorBuffer = createBuffer(colors, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);

// Uniform buffer for view-projection matrix
const vpBuffer = device.createBuffer({
  size: 64,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

// Uniform buffer for compute
const simBuffer = device.createBuffer({
  size: 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

// Pipeline setups
const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: shaderModule,
    entryPoint: 'vs_main',
    buffers: [
      {
        arrayStride: 12,
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
      },
      {
        arrayStride: 12,
        stepMode: 'instance',
        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }]
      },
      {
        arrayStride: 16,
        stepMode: 'instance',
        attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }]
      }
    ]
  },
  fragment: {
    module: shaderModule,
    entryPoint: 'fs_main',
    targets: [{ format: presentationFormat }]
  },
  primitive: {
    topology: 'triangle-list',
    cullMode: 'back'
  },
  depthStencil: {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less'
  }
});

const computePipeline = device.createComputePipeline({
  layout: 'auto',
  compute: { module: shaderModule, entryPoint: 'cs_main' }
});

const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height, 1],
  format: 'depth24plus',
  usage: GPUTextureUsage.RENDER_ATTACHMENT
});

const vpBindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: vpBuffer } }]
});

const simBindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(1),
  entries: [{ binding: 1, resource: { buffer: simBuffer } }]
});

const colorBindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: colorBuffer } }]
});

// Matrices
function mat4Perspective(out, fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  out[0] = f / aspect;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = f;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = (far) / (near - far);
  out[11] = -1;
  out[12] = 0;
  out[13] = 0;
  out[14] = (near * far) / (near - far);
  out[15] = 0;
}

function mat4LookAt(out, eye, center, up) {
  const x0 = up[1] * (eye[2] - center[2]) - up[2] * (eye[1] - center[1]);
  const x1 = up[2] * (eye[0] - center[0]) - up[0] * (eye[2] - center[2]);
  const x2 = up[0] * (eye[1] - center[1]) - up[1] * (eye[0] - center[0]);
  const l = Math.hypot(x0, x1, x2);
  const nx0 = x0 / l, nx1 = x1 / l, nx2 = x2 / l;

  const y0 = (eye[1] - center[1]) * nx2 - (eye[2] - center[2]) * nx1;
  const y1 = (eye[2] - center[2]) * nx0 - (eye[0] - center[0]) * nx2;
  const y2 = (eye[0] - center[0]) * nx1 - (eye[1] - center[1]) * nx0;
  const ny0 = y0, ny1 = y1, ny2 = y2;

  const z0 = eye[0] - center[0];
  const z1 = eye[1] - center[1];
  const z2 = eye[2] - center[2];
  const nl = Math.hypot(z0, z1, z2);
  const nz0 = z0 / nl, nz1 = z1 / nl, nz2 = z2 / nl;

  out[0] = nx0;
  out[1] = ny0;
  out[2] = nz0;
  out[3] = 0;
  out[4] = nx1;
  out[5] = ny1;
  out[6] = nz1;
  out[7] = 0;
  out[8] = nx2;
  out[9] = ny2;
  out[10] = nz2;
  out[11] = 0;
  out[12] = -(nx0 * eye[0] + nx1 * eye[1] + nx2 * eye[2]);
  out[13] = -(ny0 * eye[0] + ny1 * eye[1] + ny2 * eye[2]);
  out[14] = -(nz0 * eye[0] + nz1 * eye[1] + nz2 * eye[2]);
  out[15] = 1;
}

const vpMatrix = new Float32Array(16);
const perspective = new Float32Array(16);
const view = new Float32Array(16);

function updateMatrices(time) {
  mat4Perspective(perspective, Math.PI / 4, canvas.width / canvas.height, 0.1, 100.0);
  const r = 60;
  const eye = [Math.sin(time * 0.1) * r, 30, Math.cos(time * 0.1) * r];
  mat4LookAt(view, eye, [0,0,0], [0,1,0]);
  // multiply perspective * view
  for (let i=0;i<16;i+=4) {
    for (let j=0;j<4;j++) {
      vpMatrix[i+j] =
        perspective[i+0]*view[j+0*4] +
        perspective[i+1]*view[j+1*4] +
        perspective[i+2]*view[j+2*4] +
        perspective[i+3]*view[j+3*4];
    }
  }
  device.queue.writeBuffer(vpBuffer, 0, vpMatrix.buffer);
}

function frame(timeMs) {
  const time = timeMs / 1000;
  updateMatrices(time);
  device.queue.writeBuffer(simBuffer, 0, new Float32Array([time, numInstances]));

  const commandEncoder = device.createCommandEncoder();

  const passCompute = commandEncoder.beginComputePass();
  passCompute.setPipeline(computePipeline);
  passCompute.setBindGroup(0, colorBindGroup);
  passCompute.setBindGroup(1, simBindGroup);
  const wgCount = Math.ceil(numInstances / 64);
  passCompute.dispatchWorkgroups(wgCount);
  passCompute.end();

  const textureView = context.getCurrentTexture().createView();

  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store'
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store'
    }
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, vpBindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setVertexBuffer(1, instanceBuffer);
  pass.setVertexBuffer(2, colorBuffer);
  pass.setIndexBuffer(indexBuffer, 'uint16');
  pass.drawIndexed(cubeIndices.length, numInstances);
  pass.end();

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
