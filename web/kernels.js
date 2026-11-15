// WGSL kernels of the slot detector's forward pass, and the helpers that bind them.
//
// Activations are row-major f32 matrices in storage buffers; a kernel addresses a matrix
// by a first row, a row stride and (for outputs) a first column, so that one buffer can
// hold several tensors and padded rows. Quantised weights are read in the packed format
// of `src/crescent_moon/pack.py`. Every kernel takes its sizes src a uniform block of
// u32 fields, listed in FIELDS under the kernel's name in the order the shader declares them.

const GROUP = 32; // weights per scale
const FIELDS = {};

const GELU = `
fn gelu(v: vec4<f32>) -> vec4<f32> {
  // tanh approximation, as torch.nn.functional.gelu(approximate="tanh"). tanh is 1 in
  // f32 beyond 10, and some implementations return NaN for large arguments.
  let t = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), vec4<f32>(-10.0), vec4<f32>(10.0));
  return 0.5 * v * (1.0 + tanh(t));
}`;

// Offsets of the four rows one invocation handles; rows past the end repeat the last
// row and are not written.
const ROWS4 = `
  let last = p.rows - 1u;
  let r = vec4<u32>(min(row, last), min(row + 1u, last), min(row + 2u, last), min(row + 3u, last));
  let xr = (r + p.x_row0) * p.x_stride;`;

const WRITE4 = (value) => `
  for (var k = 0u; k < 4u; k++) {
    if (row + k < p.rows) {
      let at = (p.y_row0 + row + k) * p.y_stride + p.y_col0 + o;
      ${value}
    }
  }`;

// y[row, o] = bias[o] + sum_g scale[o, g] * sum_j q[o, 32 g + j] * x[row, 32 g + j],
// for four rows per invocation so that each weight is decoded once for four products.
// flags: 1 adds to y, 2 applies GELU.
export function quantisedLinear(bits) {
  const perWord = 32 / bits, words = GROUP / perWord, shift = 32 - bits;
  const taps = [];
  for (let j = 0; j < perWord; j++) {
    const up = shift - bits * j;
    const field = up ? `i32(word << ${up}u) >> ${shift}u` : `i32(word) >> ${shift}u`;
    taps.push(`      acc += f32(${field}) * vec4<f32>(x[xr.x + c + ${j}u], x[xr.y + c + ${j}u], `
      + `x[xr.z + c + ${j}u], x[xr.w + c + ${j}u]);`);
  }
  return `
struct P { rows: u32, x_row0: u32, x_stride: u32, groups: u32,
           outs: u32, y_row0: u32, y_stride: u32, y_col0: u32,
           scale_stride: u32, flags: u32, out0: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> s: array<u32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
${GELU}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let o = id.x + p.out0;
  let row = id.y * 4u;
  if (o >= p.outs || row >= p.rows) { return; }
${ROWS4}
  var total = vec4<f32>(0.0);
  var wi = o * p.groups * ${words}u;
  for (var g = 0u; g < p.groups; g++) {
    var acc = vec4<f32>(0.0);
    for (var k = 0u; k < ${words}u; k++) {
      let word = w[wi];
      wi++;
      let c = g * ${GROUP}u + k * ${perWord}u;
${taps.join("\n")}
    }
    let si = o * p.scale_stride + g;
    total += unpack2x16float(s[si >> 1u])[si & 1u] * acc;
  }
  total += b[o];
  if ((p.flags & 2u) != 0u) { total = gelu(total); }
${WRITE4("if ((p.flags & 1u) != 0u) { y[at] += total[k]; } else { y[at] = total[k]; }")}
}`;
}
FIELDS.quantisedLinear = ["rows", "x_row0", "x_stride", "groups", "outs", "y_row0", "y_stride",
  "y_col0", "scale_stride", "flags", "out0", "pad"];

// y[row, o] (+)= sum_i w[o, i] * x[row, i] with dense FP16 weights [outs, ins], ins even.
// flags: 1 adds to y.
export const halfLinear = `
struct P { rows: u32, x_row0: u32, x_stride: u32, ins: u32,
           outs: u32, y_row0: u32, y_stride: u32, y_col0: u32, flags: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let o = id.x;
  let row = id.y * 4u;
  if (o >= p.outs || row >= p.rows) { return; }
${ROWS4}
  var total = vec4<f32>(0.0);
  let half = p.ins / 2u;
  let base = o * half;
  for (var i = 0u; i < half; i++) {
    let pair = unpack2x16float(w[base + i]);
    let c = 2u * i;
    total += pair.x * vec4<f32>(x[xr.x + c], x[xr.y + c], x[xr.z + c], x[xr.w + c]);
    total += pair.y * vec4<f32>(x[xr.x + c + 1u], x[xr.y + c + 1u], x[xr.z + c + 1u], x[xr.w + c + 1u]);
  }
${WRITE4("if ((p.flags & 1u) != 0u) { y[at] += total[k]; } else { y[at] = total[k]; }")}
}`;
FIELDS.halfLinear = ["rows", "x_row0", "x_stride", "ins", "outs", "y_row0", "y_stride", "y_col0",
  "flags"];

// y[row] = (x[row] - mean) / sqrt(variance + 1e-5) * weight + bias over `dim` columns.
export const layerNorm = `
struct P { rows: u32, x_row0: u32, x_stride: u32, dim: u32, y_row0: u32, y_stride: u32, y_col0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.rows) { return; }
  let src = (p.x_row0 + id.x) * p.x_stride;
  var sum = 0.0;
  for (var i = 0u; i < p.dim; i++) { sum += x[src + i]; }
  let mean = sum / f32(p.dim);
  var squares = 0.0;
  for (var i = 0u; i < p.dim; i++) { let d = x[src + i] - mean; squares += d * d; }
  let scale = inverseSqrt(squares / f32(p.dim) + 1e-5);
  let to = (p.y_row0 + id.x) * p.y_stride + p.y_col0;
  for (var i = 0u; i < p.dim; i++) { y[to + i] = (x[src + i] - mean) * scale * weight[i] + bias[i]; }
}`;
FIELDS.layerNorm = ["rows", "x_row0", "x_stride", "dim", "y_row0", "y_stride", "y_col0"];

// GELU in place over `count` values src `first`; the grid is `width` invocations wide.
export const geluInPlace = `
struct P { first: u32, count: u32, width: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
${GELU}
@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.y * p.width + id.x;
  if (id.x >= p.width || i >= p.count) { return; }
  y[p.first + i] = gelu(vec4<f32>(y[p.first + i])).x;
}`;
FIELDS.geluInPlace = ["first", "count", "width"];

// Rotary position embedding in place on the q and k parts of a fused qkv matrix
// [rows, 3 dim]: in every head the first 16 columns are real parts and the next 16
// imaginary parts, turned by the angle of (row position, frequency). `table` holds
// (cos, sin) per position and frequency.
export const rotary = `
struct P { rows: u32, stride: u32, dim: u32, heads: u32, position0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> qkv: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  let row = id.y;
  if (head >= p.heads || row >= p.rows) { return; }
  let width = p.dim / p.heads;
  let angles = (p.position0 + row) * 32u;
  for (var part = 0u; part < 2u; part++) {
    let src = row * p.stride + part * p.dim + head * width;
    for (var j = 0u; j < 16u; j++) {
      let c = table[angles + 2u * j];
      let s = table[angles + 2u * j + 1u];
      let re = qkv[src + j];
      let im = qkv[src + 16u + j];
      qkv[src + j] = re * c - im * s;
      qkv[src + 16u + j] = re * s + im * c;
    }
  }
}`;
FIELDS.rotary = ["rows", "stride", "dim", "heads", "position0"];

// Attention over a fused qkv matrix [rows, 3 dim] (q, then k, then v; heads side by
// side): y[row, head] = softmax_n(q . k_n / sqrt(width)) v_n. Rows before `prefix`
// attend to the first `prefix` rows, the others to all rows. The softmax is computed
// online, rescaling the running sum whenever a larger score appears.
export function attention(headWidth) {
  const quads = headWidth / 4;
  return `
struct P { rows: u32, stride: u32, dim: u32, heads: u32, prefix: u32, y_stride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn load(at: u32) -> vec4<f32> { return vec4<f32>(qkv[at], qkv[at + 1u], qkv[at + 2u], qkv[at + 3u]); }
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  let row = id.y;
  if (head >= p.heads || row >= p.rows) { return; }
  let keys = select(p.rows, p.prefix, row < p.prefix);
  let column = head * ${headWidth}u;
  var q: array<vec4<f32>, ${quads}>;
  var acc: array<vec4<f32>, ${quads}>;
  for (var t = 0u; t < ${quads}u; t++) {
    q[t] = load(row * p.stride + column + 4u * t) * ${1 / Math.sqrt(headWidth)};
    acc[t] = vec4<f32>(0.0);
  }
  var top = -3.0e38;
  var sum = 0.0;
  for (var n = 0u; n < keys; n++) {
    let k = n * p.stride + p.dim + column;
    var score = 0.0;
    for (var t = 0u; t < ${quads}u; t++) { score += dot(q[t], load(k + 4u * t)); }
    if (score > top) {
      let shrink = exp(top - score);
      sum *= shrink;
      for (var t = 0u; t < ${quads}u; t++) { acc[t] *= shrink; }
      top = score;
    }
    let weight = exp(score - top);
    sum += weight;
    let v = k + p.dim;
    for (var t = 0u; t < ${quads}u; t++) { acc[t] += weight * load(v + 4u * t); }
  }
  let to = row * p.y_stride + column;
  for (var t = 0u; t < ${quads}u; t++) {
    let out = acc[t] / sum;
    y[to + 4u * t] = out.x; y[to + 4u * t + 1u] = out.y;
    y[to + 4u * t + 2u] = out.z; y[to + 4u * t + 3u] = out.w;
  }
}`;
}
FIELDS.attention = ["rows", "stride", "dim", "heads", "prefix", "y_stride"];

// For each row and each of `segments` runs of `width` logits: the index of the largest
// logit and its softmax probability, written as two f32 at y[(y0 + row * segments + segment) * 2].
export const argmax = `
struct P { rows: u32, x_stride: u32, segments: u32, width: u32, y0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let segment = id.x;
  let row = id.y;
  if (segment >= p.segments || row >= p.rows) { return; }
  let src = row * p.x_stride + segment * p.width;
  var best = 0u;
  var top = x[src];
  for (var i = 1u; i < p.width; i++) {
    let v = x[src + i];
    if (v > top) { top = v; best = i; }
  }
  var sum = 0.0;
  for (var i = 0u; i < p.width; i++) { sum += exp(x[src + i] - top); }
  let to = (p.y0 + row * p.segments + segment) * 2u;
  y[to] = f32(best);
  y[to + 1u] = 1.0 / sum;
}`;
FIELDS.argmax = ["rows", "x_stride", "segments", "width", "y0"];

// Compiles kernels once per device and builds the steps of a pass: each step is a
// pipeline, a bind group holding its uniform block and buffers, and a dispatch size.
export class Kernels {
  constructor(device) {
    this.device = device;
    this.pipelines = new Map();
  }

  pipeline(key, source) {
    if (!this.pipelines.has(key)) {
      const module = this.device.createShaderModule({ code: source, label: key });
      this.pipelines.set(key, this.device.createComputePipeline({
        label: key, layout: "auto", compute: { module, entryPoint: "main" },
      }));
    }
    return this.pipelines.get(key);
  }

  step(key, source, fields, values, buffers, dispatch) {
    const data = new Uint32Array(Math.max(4, Math.ceil(fields.length / 4) * 4));
    fields.forEach((name, i) => {
      const value = values[name] ?? 0;
      if (!Number.isInteger(value) || value < 0) throw new Error(`${key}: ${name} = ${value}`);
      data[i] = value;
    });
    const uniform = this.device.createBuffer({
      size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniform, 0, data);
    const pipeline = this.pipeline(key, source);
    const entries = [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } }));
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    return { pipeline, group, dispatch, label: key };
  }

  // v: { rows, xRow0, xStride, groups, outs, yRow0, yStride, yCol0, scaleStride, add, gelu }
  quantisedLinear(bits, x, layer, y, v) {
    const values = {
      rows: v.rows, x_row0: v.xRow0 ?? 0, x_stride: v.xStride, groups: layer.groups,
      outs: layer.rows, y_row0: v.yRow0 ?? 0, y_stride: v.yStride, y_col0: v.yCol0 ?? 0,
      scale_stride: layer.scaleStride, flags: (v.add ? 1 : 0) | (v.gelu ? 2 : 0),
    };
    return this.step(`quantised-linear-${bits}`, quantisedLinear(bits), FIELDS.quantisedLinear,
      values, [x, layer.weight, layer.scale, layer.bias, y],
      [Math.ceil(layer.rows / 8), Math.ceil(Math.ceil(v.rows / 4) / 8)]);
  }

  halfLinear(x, weight, y, v) {
    const values = {
      rows: v.rows, x_row0: v.xRow0 ?? 0, x_stride: v.xStride, ins: v.ins, outs: v.outs,
      y_row0: v.yRow0 ?? 0, y_stride: v.yStride, y_col0: v.yCol0 ?? 0, flags: v.add ? 1 : 0,
    };
    return this.step("half-linear", halfLinear, FIELDS.halfLinear, values, [x, weight, y],
      [Math.ceil(v.outs / 8), Math.ceil(Math.ceil(v.rows / 4) / 8)]);
  }

  layerNorm(x, weight, bias, y, v) {
    const values = {
      rows: v.rows, x_row0: v.xRow0 ?? 0, x_stride: v.xStride, dim: v.dim,
      y_row0: v.yRow0 ?? 0, y_stride: v.yStride, y_col0: v.yCol0 ?? 0,
    };
    return this.step("layer-norm", layerNorm, FIELDS.layerNorm, values, [x, weight, bias, y],
      [Math.ceil(v.rows / 64)]);
  }

  gelu(y, first, count) {
    const width = 16384;
    return this.step("gelu", geluInPlace, FIELDS.geluInPlace, { first, count, width }, [y],
      [width / 64, Math.ceil(Math.ceil(count / width) / 4)]);
  }

  rotary(table, qkv, v) {
    const values = { rows: v.rows, stride: v.stride, dim: v.dim, heads: v.heads, position0: v.position0 ?? 0 };
    return this.step("rotary", rotary, FIELDS.rotary, values, [table, qkv],
      [Math.ceil(v.heads / 8), Math.ceil(v.rows / 8)]);
  }

  attention(qkv, y, v) {
    const width = v.dim / v.heads;
    const values = {
      rows: v.rows, stride: v.stride, dim: v.dim, heads: v.heads, prefix: v.prefix ?? 0,
      y_stride: v.yStride,
    };
    return this.step(`attention-${width}`, attention(width), FIELDS.attention, values, [qkv, y],
      [Math.ceil(v.heads / 8), Math.ceil(v.rows / 8)]);
  }

  argmax(x, y, v) {
    const values = { rows: v.rows, x_stride: v.xStride, segments: v.segments, width: v.width, y0: v.y0 ?? 0 };
    return this.step("argmax", argmax, FIELDS.argmax, values, [x, y],
      [Math.ceil(v.segments / 4), Math.ceil(v.rows / 4)]);
  }
}

// Runs steps in order. A step is a kernel step, or { copy: [source, sourceOffset,
// destination, destinationOffset, bytes] }.
export function encode(device, steps) {
  const encoder = device.createCommandEncoder();
  let pass = null;
  for (const step of steps) {
    if (step.copy) {
      if (pass) { pass.end(); pass = null; }
      encoder.copyBufferToBuffer(...step.copy);
      continue;
    }
    pass ??= encoder.beginComputePass();
    pass.setPipeline(step.pipeline);
    pass.setBindGroup(0, step.group);
    pass.dispatchWorkgroups(...step.dispatch);
  }
  if (pass) pass.end();
  return encoder.finish();
}

export async function read(device, buffer, offset, bytes) {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, offset, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return out;
}

export function storage(device, bytes, extra = 0) {
  return device.createBuffer({
    size: Math.ceil(bytes / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra,
  });
}

export function upload(device, array) {
  const buffer = storage(device, array.byteLength);
  device.queue.writeBuffer(buffer, 0, array.buffer, array.byteOffset, array.byteLength);
  return buffer;
}
