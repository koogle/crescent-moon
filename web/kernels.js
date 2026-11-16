// WGSL kernels of the slot detector's forward pass, and the helpers that bind them.
//
// Activations are row-major f32 matrices in storage buffers; a kernel addresses a matrix
// by a first row, a row stride and (for outputs) a first column, so that one buffer can
// hold several tensors and padded rows. Quantised weights are read in the packed format
// of `src/crescent_moon/pack.py`. Every kernel takes its sizes src a uniform block of
// u32 fields, listed in FIELDS under the kernel's name in the order the shader declares them.

const GROUP = 32; // weights per scale
// Work per invocation: the tile of the quantised product (outputs by token rows; rows
// a multiple of 4) and the query rows of one head in attention. TUNING is the fastest
// on an NVIDIA L4; CHOICES are what `tune` in model.js times on the device in use.
export const TUNING = { outs: 4, rows: 16, queries: 2 };
export const CHOICES = {
  tiles: [[4, 16], [4, 8], [8, 8], [2, 16], [4, 4], [2, 8]],
  queries: [2, 1, 4],
};
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

export function quantisedLinear(bits, OUTS = TUNING.outs, ROWS = TUNING.rows) {
  // A word holds 32 / bits weights; they are decoded four at a time into a vec4 and
  // multiplied with four columns of a row in one dot product. One invocation computes a
  // tile of OUTS outputs by ROWS rows, so each input vec4 is loaded once for OUTS outputs
  // and each weight word is decoded once for ROWS rows.
  const perWord = 32 / bits, words = GROUP / perWord, quads = perWord / 4, shift = 32 - bits;
  const outs = [...Array(OUTS).keys()], fours = [...Array(ROWS / 4).keys()], qs = [...Array(quads).keys()];
  const lines = (items, make) => items.map(make).join("\n");
  const ups = (q) => [0, 1, 2, 3].map((j) => `${shift - bits * (4 * q + j)}u`).join(", ");
  const sum = (i, h) => `vec4<f32>(${[0, 1, 2, 3].map((j) =>
    qs.map((q) => `dot(d${i}_${q}, x${4 * h + j}_${q})`).join(" + ")).join(", ")})`;
  return `
struct P { rows: u32, x_row0: u32, x_stride: u32, groups: u32,
           outs: u32, y_row0: u32, y_stride: u32, y_col0: u32,
           scale_stride: u32, flags: u32, out0: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> s: array<u32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
${GELU}
fn put(row: u32, o: u32, value: vec4<f32>) {
  if (o >= p.outs) { return; }
  let v = select(value, gelu(value), (p.flags & 2u) != 0u);
  for (var k = 0u; k < 4u; k++) {
    if (row + k < p.rows) {
      let at = (p.y_row0 + row + k) * p.y_stride + p.y_col0 + o;
      if ((p.flags & 1u) != 0u) { y[at] += v[k]; } else { y[at] = v[k]; }
    }
  }
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let o = id.x * ${OUTS}u + p.out0;
  let row = id.y * ${ROWS}u;
  if (o >= p.outs || row >= p.rows) { return; }
  // Offsets, in vec4s, of the rows; rows and outputs past the end repeat the last and
  // are not written.
  let last = p.rows - 1u;
  let quarter = p.x_stride >> 2u;
${lines([...Array(ROWS).keys()], (r) => `  let r${r} = (min(row + ${r}u, last) + p.x_row0) * quarter;`)}
${lines(outs, (i) => `  let o${i} = min(o + ${i}u, p.outs - 1u);
  var w${i} = o${i} * p.groups * ${words}u;
${lines(fours, (h) => `  var t${i}_${h} = vec4<f32>(0.0);`)}`)}
  for (var g = 0u; g < p.groups; g++) {
${lines(outs, (i) => lines(fours, (h) => `    var a${i}_${h} = vec4<f32>(0.0);`))}
    for (var k = 0u; k < ${words}u; k++) {
      let c = g * ${GROUP / 4}u + k * ${quads}u;
${lines([...Array(ROWS).keys()], (r) => lines(qs, (q) => `      let x${r}_${q} = x[r${r} + c + ${q}u];`))}
${lines(outs, (i) => `      let word${i} = w[w${i} + k];
${lines(qs, (q) => `      let d${i}_${q} = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(word${i}) << vec4<u32>(${ups(q)})) >> vec4<u32>(${shift}u));`)}
${lines(fours, (h) => `      a${i}_${h} += ${sum(i, h)};`)}`)}
    }
${lines(outs, (i) => `    w${i} += ${words}u;
    let s${i} = o${i} * p.scale_stride + g;
    let scale${i} = unpack2x16float(s[s${i} >> 1u])[s${i} & 1u];
${lines(fours, (h) => `    t${i}_${h} += scale${i} * a${i}_${h};`)}`)}
  }
${lines(outs, (i) => lines(fours, (h) => `  put(row + ${4 * h}u, o + ${i}u, t${i}_${h} + b[o${i}]);`))}
}`;
}

// y[row, o] = bias[o] + sum_g scale[o, g] * sum_j q[o, 32 g + j] * x[row, 32 g + j],
// x is read as vec4s: its row stride is a multiple of 4. flags: 1 adds to y, 2 applies GELU.
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

// y[row] = (x[row] - mean) / sqrt(variance + 1e-5) * weight + bias over `dim` columns,
// four columns at a time: strides, `dim` and `y_col0` are multiples of 4.
export const layerNorm = `
struct P { rows: u32, x_row0: u32, x_stride: u32, dim: u32, y_row0: u32, y_stride: u32, y_col0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.rows) { return; }
  let src = ((p.x_row0 + id.x) * p.x_stride) >> 2u;
  let n = p.dim >> 2u;
  var sum = vec4<f32>(0.0);
  for (var i = 0u; i < n; i++) { sum += x[src + i]; }
  let mean = (sum.x + sum.y + sum.z + sum.w) / f32(p.dim);
  var squares = vec4<f32>(0.0);
  for (var i = 0u; i < n; i++) { let d = x[src + i] - mean; squares += d * d; }
  let scale = inverseSqrt((squares.x + squares.y + squares.z + squares.w) / f32(p.dim) + 1e-5);
  let to = ((p.y_row0 + id.x) * p.y_stride + p.y_col0) >> 2u;
  for (var i = 0u; i < n; i++) { y[to + i] = (x[src + i] - mean) * scale * weight[i] + bias[i]; }
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
export function attention(headWidth, QUERIES = TUNING.queries) {
  const quads = headWidth / 4;
  return `
struct P { rows: u32, stride: u32, dim: u32, heads: u32, prefix: u32, y_stride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
// Rows and heads start at multiples of four floats, so they are read and written as vec4s.
fn load(at: u32) -> vec4<f32> { return qkv[at >> 2u]; }
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  let row = id.y * ${QUERIES}u;
  if (head >= p.heads || row >= p.rows) { return; }
  let column = head * ${headWidth}u;
  // QUERIES rows of one head share every key and value load. A row past the end repeats
  // the last and is not written.
  var q: array<vec4<f32>, ${QUERIES * quads}>;
  var acc: array<vec4<f32>, ${QUERIES * quads}>;
  var keys: array<u32, ${QUERIES}>;
  var top: array<f32, ${QUERIES}>;
  var sum: array<f32, ${QUERIES}>;
  var most = 0u;
  for (var i = 0u; i < ${QUERIES}u; i++) {
    let r = min(row + i, p.rows - 1u);
    keys[i] = select(p.rows, p.prefix, r < p.prefix);
    most = max(most, keys[i]);
    top[i] = -3.0e38;
    sum[i] = 0.0;
    for (var t = 0u; t < ${quads}u; t++) {
      q[i * ${quads}u + t] = load(r * p.stride + column + 4u * t) * ${1 / Math.sqrt(headWidth)};
      acc[i * ${quads}u + t] = vec4<f32>(0.0);
    }
  }
  var key: array<vec4<f32>, ${quads}>;
  var weight: array<f32, ${QUERIES}>;
  for (var n = 0u; n < most; n++) {
    let k = n * p.stride + p.dim + column;
    for (var t = 0u; t < ${quads}u; t++) { key[t] = load(k + 4u * t); }
    for (var i = 0u; i < ${QUERIES}u; i++) {
      weight[i] = 0.0;
      if (n < keys[i]) {
        let base = i * ${quads}u;
        var score = 0.0;
        for (var t = 0u; t < ${quads}u; t++) { score += dot(q[base + t], key[t]); }
        if (score > top[i]) {
          let shrink = exp(top[i] - score);
          sum[i] *= shrink;
          for (var t = 0u; t < ${quads}u; t++) { acc[base + t] *= shrink; }
          top[i] = score;
        }
        weight[i] = exp(score - top[i]);
        sum[i] += weight[i];
      }
    }
    let v = k + p.dim;
    for (var t = 0u; t < ${quads}u; t++) {
      let value = load(v + 4u * t);
      for (var i = 0u; i < ${QUERIES}u; i++) { acc[i * ${quads}u + t] += weight[i] * value; }
    }
  }
  for (var i = 0u; i < ${QUERIES}u; i++) {
    if (row + i < p.rows) {
      let to = ((row + i) * p.y_stride + column) >> 2u;
      for (var t = 0u; t < ${quads}u; t++) { y[to + t] = acc[i * ${quads}u + t] / sum[i]; }
    }
  }
}`;
}
FIELDS.attention = ["rows", "stride", "dim", "heads", "prefix", "y_stride"];

// y[y_row0 + cell] = mean of the rows of x in that pool x pool cell of a side x side grid
// of rows (the last cell of a row or column is smaller); ceil(side / pool) ** 2 cells.
export const poolGrid = `
struct P { side: u32, pool: u32, x_stride: u32, dim: u32, y_row0: u32, y_stride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cells = (p.side + p.pool - 1u) / p.pool;
  if (id.x >= (p.dim >> 2u) || id.y >= cells * cells) { return; }
  let top = (id.y / cells) * p.pool;
  let left = (id.y % cells) * p.pool;
  var sum = vec4<f32>(0.0);
  var count = 0.0;
  for (var dy = 0u; dy < p.pool; dy++) {
    for (var dx = 0u; dx < p.pool; dx++) {
      if (top + dy < p.side && left + dx < p.side) {
        sum += x[((top + dy) * p.side + left + dx) * (p.x_stride >> 2u) + id.x];
        count += 1.0;
      }
    }
  }
  y[(p.y_row0 + id.y) * (p.y_stride >> 2u) + id.x] = sum / count;
}`;
FIELDS.poolGrid = ["side", "pool", "x_stride", "dim", "y_row0", "y_stride"];

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
  // `tuning`: { outs, rows, queries }, see TUNING.
  constructor(device, tuning = TUNING) {
    this.device = device;
    this.tuning = tuning;
    this.pipelines = new Map();
    this.modules = new Map();
  }

  pipeline(key, source) {
    if (!this.pipelines.has(key)) {
      const module = this.device.createShaderModule({ code: source, label: key });
      this.modules.set(key, module);
      this.pipelines.set(key, this.device.createComputePipeline({
        label: key, layout: "auto", compute: { module, entryPoint: "main" },
      }));
    }
    return this.pipelines.get(key);
  }

  // Compilation errors of the kernels used so far, as "kernel: message (line)".
  async problems() {
    const found = [];
    for (const [key, module] of this.modules) {
      for (const message of (await module.getCompilationInfo()).messages) {
        if (message.type === "error") found.push(`${key}: ${message.message} (line ${message.lineNum})`);
      }
    }
    return found;
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
    if (v.xStride % 4) throw new Error(`The quantised product reads rows of a multiple of 4 columns, not ${v.xStride}`);
    const values = {
      rows: v.rows, x_row0: v.xRow0 ?? 0, x_stride: v.xStride, groups: layer.groups,
      outs: layer.rows, y_row0: v.yRow0 ?? 0, y_stride: v.yStride, y_col0: v.yCol0 ?? 0,
      scale_stride: layer.scaleStride, flags: (v.add ? 1 : 0) | (v.gelu ? 2 : 0),
    };
    const { outs, rows } = this.tuning;
    return this.step(`quantised-linear-${bits}`, quantisedLinear(bits, outs, rows),
      FIELDS.quantisedLinear, values, [x, layer.weight, layer.scale, layer.bias, y],
      [Math.ceil(Math.ceil(layer.rows / outs) / 8), Math.ceil(Math.ceil(v.rows / rows) / 8)]);
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
    if ([v.xStride, v.dim, v.yStride, v.yCol0 ?? 0].some((n) => n % 4)) {
      throw new Error("Layer norm reads and writes four columns at a time");
    }
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
    if (v.stride % 4 || v.yStride % 4 || width % 4) throw new Error("Attention reads and writes rows and heads of a multiple of 4 floats");
    const values = {
      rows: v.rows, stride: v.stride, dim: v.dim, heads: v.heads, prefix: v.prefix ?? 0,
      y_stride: v.yStride,
    };
    const { queries } = this.tuning;
    return this.step(`attention-${width}`, attention(width, queries), FIELDS.attention, values,
      [qkv, y], [Math.ceil(v.heads / 8), Math.ceil(Math.ceil(v.rows / queries) / 8)]);
  }

  // v: { side, pool, xStride, dim, yRow0, yStride }
  poolGrid(x, y, v) {
    const cells = Math.ceil(v.side / v.pool) ** 2;
    const values = { side: v.side, pool: v.pool, x_stride: v.xStride, dim: v.dim, y_row0: v.yRow0 ?? 0, y_stride: v.yStride };
    return this.step("pool-grid", poolGrid, FIELDS.poolGrid, values, [x, y],
      [Math.ceil(v.dim / 4 / 8), Math.ceil(cells / 8)]);
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

// Seconds on the GPU per kernel label for `steps`, each in its own pass between two
// timestamps; the device needs the timestamp-query feature.
export async function profile(device, steps) {
  const kernels = steps.filter((step) => !step.copy);
  const queries = device.createQuerySet({ type: "timestamp", count: 2 * kernels.length });
  const bytes = 16 * kernels.length;
  const resolved = device.createBuffer({ size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  let n = 0;
  for (const step of steps) {
    if (step.copy) { encoder.copyBufferToBuffer(...step.copy); continue; }
    const pass = encoder.beginComputePass({
      timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 2 * n, endOfPassWriteIndex: 2 * n + 1 },
    });
    pass.setPipeline(step.pipeline);
    pass.setBindGroup(0, step.group);
    pass.dispatchWorkgroups(...step.dispatch);
    pass.end();
    n++;
  }
  encoder.resolveQuerySet(queries, 0, 2 * kernels.length, resolved, 0);
  encoder.copyBufferToBuffer(resolved, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const ticks = new BigUint64Array(staging.getMappedRange().slice(0));
  staging.destroy(); resolved.destroy(); queries.destroy();
  const seconds = {};
  kernels.forEach((step, i) => {
    seconds[step.label] = (seconds[step.label] ?? 0) + Number(ticks[2 * i + 1] - ticks[2 * i]) / 1e9;
  });
  return seconds;
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
