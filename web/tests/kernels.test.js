// Each kernel against a plain JavaScript computation of the same arithmetic on random
// inputs, including padded strides, row and column offsets and the add and GELU flags.

import { CHOICES, Kernels, TUNING, encode, read, upload, storage } from "../kernels.js";

function random(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function toHalf(value) {
  // Exact for the values the tests use (small multiples of powers of two).
  if (value === 0) return 0;
  const sign = value < 0 ? 0x8000 : 0, v = Math.abs(value);
  const exponent = Math.floor(Math.log2(v));
  const fraction = Math.round((v / 2 ** exponent - 1) * 1024);
  return sign | ((exponent + 15) << 10) | fraction;
}

const gelu = (v) => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)));
const floats = (rand, n, scale = 1) => Float32Array.from({ length: n }, () => (rand() * 2 - 1) * scale);

function worst(got, want) {
  let error = 0, size = 0;
  for (let i = 0; i < want.length; i++) {
    error = Math.max(error, Math.abs(got[i] - want[i]));
    size = Math.max(size, Math.abs(want[i]));
  }
  return error / Math.max(size, 1e-9);
}

async function quantised(device, kernels, bits, { rows, columns, outs, add, activation, xRow0, yRow0, yCol0 }) {
  const rand = random(bits * 1000 + rows + columns + outs);
  const padded = Math.ceil(columns / 32) * 32, groups = padded / 32, perWord = 32 / bits;
  const scaleStride = groups + (groups % 2);
  const top = 2 ** (bits - 1);
  const q = new Int32Array(outs * padded), scale = new Float32Array(outs * groups);
  const words = new Uint32Array(outs * padded / perWord), halves = new Uint16Array(outs * scaleStride);
  for (let o = 0; o < outs; o++) {
    for (let c = 0; c < columns; c++) {
      const v = Math.floor(rand() * 2 * top) - top;
      q[o * padded + c] = v;
      const at = o * padded + c;
      words[Math.floor(at / perWord)] |= ((v & (2 ** bits - 1)) << (bits * (at % perWord))) >>> 0;
    }
    for (let g = 0; g < groups; g++) {
      scale[o * groups + g] = (1 + Math.floor(rand() * 30)) / 256;
      halves[o * scaleStride + g] = toHalf(scale[o * groups + g]);
    }
  }
  const xStride = padded + 32, yStride = outs + 5 + yCol0;
  const x = floats(rand, (xRow0 + rows) * xStride), bias = floats(rand, outs);
  if (activation) { bias[0] = 300; bias[1] = -300; } // far past f32's tanh range
  for (let r = 0; r < xRow0 + rows; r++) for (let c = columns; c < padded; c++) x[r * xStride + c] = 0;
  const y0 = floats(rand, (yRow0 + rows) * yStride);
  const want = y0.slice();
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outs; o++) {
      let total = 0;
      for (let g = 0; g < groups; g++) {
        let acc = 0;
        for (let j = 0; j < 32; j++) acc += q[o * padded + g * 32 + j] * x[(xRow0 + r) * xStride + g * 32 + j];
        total += scale[o * groups + g] * acc;
      }
      total += bias[o];
      if (activation) total = gelu(total);
      const at = (yRow0 + r) * yStride + yCol0 + o;
      want[at] = add ? want[at] + total : total;
    }
  }
  const layer = {
    rows: outs, groups, scaleStride, weight: upload(device, words), scale: upload(device, new Uint8Array(halves.buffer)),
    bias: upload(device, bias),
  };
  const yBuffer = upload(device, y0);
  const step = kernels.quantisedLinear(bits, upload(device, x), layer, yBuffer,
    { rows, xRow0, xStride, yRow0, yStride, yCol0, add, gelu: activation });
  device.queue.submit([encode(device, [step])]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

async function half(device, kernels, { rows, ins, outs, add, xRow0, yRow0 }) {
  const rand = random(rows * 7 + ins + outs);
  const w = Float32Array.from({ length: outs * ins }, () => (Math.floor(rand() * 512) - 256) / 256);
  const halves = Uint16Array.from(w, toHalf);
  const xStride = ins + 3, yStride = outs + 2;
  const x = floats(rand, (xRow0 + rows) * xStride), y0 = floats(rand, (yRow0 + rows) * yStride);
  const want = y0.slice();
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outs; o++) {
      let total = 0;
      for (let i = 0; i < ins; i++) total += w[o * ins + i] * x[(xRow0 + r) * xStride + i];
      const at = (yRow0 + r) * yStride + o;
      want[at] = add ? want[at] + total : total;
    }
  }
  const yBuffer = upload(device, y0);
  const step = kernels.halfLinear(upload(device, x), upload(device, new Uint8Array(halves.buffer)), yBuffer,
    { rows, xRow0, xStride, ins, outs, yRow0, yStride, add });
  device.queue.submit([encode(device, [step])]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

async function norm(device, kernels, { rows, dim, yCol0 }) {
  const rand = random(rows + dim);
  const xStride = dim + 4, yStride = dim + yCol0 + 4;
  const x = floats(rand, (1 + rows) * xStride, 3), weight = floats(rand, dim), bias = floats(rand, dim);
  const y0 = floats(rand, (2 + rows) * yStride), want = y0.slice();
  for (let r = 0; r < rows; r++) {
    const row = x.subarray((1 + r) * xStride, (1 + r) * xStride + dim);
    const mean = row.reduce((a, v) => a + v, 0) / dim;
    const variance = row.reduce((a, v) => a + (v - mean) ** 2, 0) / dim;
    for (let i = 0; i < dim; i++) {
      want[(2 + r) * yStride + yCol0 + i] = (row[i] - mean) / Math.sqrt(variance + 1e-5) * weight[i] + bias[i];
    }
  }
  const yBuffer = upload(device, y0);
  const step = kernels.layerNorm(upload(device, x), upload(device, weight), upload(device, bias), yBuffer,
    { rows, xRow0: 1, xStride, dim, yRow0: 2, yStride, yCol0 });
  device.queue.submit([encode(device, [step])]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

async function geluKernel(device, kernels, { first, count }) {
  const rand = random(count);
  const y0 = floats(rand, first + count + 9, 4), want = y0.slice();
  // Magnitudes at which the cubic inside tanh is far past f32's tanh range.
  [-300, -40, 40, 300].forEach((v, i) => { y0[first + i] = v; });
  for (let i = 0; i < count; i++) want[first + i] = gelu(y0[first + i]);
  const yBuffer = upload(device, y0);
  device.queue.submit([encode(device, [kernels.gelu(yBuffer, first, count)])]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

function turn(qkv, table, { rows, stride, dim, heads, position0 }) {
  const out = qkv.slice(), width = dim / heads;
  for (let r = 0; r < rows; r++) for (let part = 0; part < 2; part++) for (let h = 0; h < heads; h++) {
    const from = r * stride + part * dim + h * width;
    for (let j = 0; j < 16; j++) {
      const c = table[(position0 + r) * 32 + 2 * j], s = table[(position0 + r) * 32 + 2 * j + 1];
      const re = qkv[from + j], im = qkv[from + 16 + j];
      out[from + j] = re * c - im * s;
      out[from + 16 + j] = re * s + im * c;
    }
  }
  return out;
}

async function rotaryKernel(device, kernels, v) {
  const rand = random(v.rows * v.heads);
  const table = floats(rand, (v.position0 + v.rows) * 32), qkv = floats(rand, v.rows * v.stride);
  const buffer = upload(device, qkv);
  device.queue.submit([encode(device, [kernels.rotary(upload(device, table), buffer, v)])]);
  return worst(await read(device, buffer, 0, qkv.byteLength), turn(qkv, table, v));
}

async function attentionKernel(device, kernels, { rows, dim, heads, prefix }) {
  const rand = random(rows + dim + prefix), width = dim / heads, stride = 3 * dim;
  const qkv = floats(rand, rows * stride, 2), yStride = dim + 4;
  const y0 = floats(rand, rows * yStride), want = y0.slice();
  for (let r = 0; r < rows; r++) for (let h = 0; h < heads; h++) {
    const keys = r < prefix ? prefix : rows;
    const scores = [];
    for (let n = 0; n < keys; n++) {
      let s = 0;
      for (let i = 0; i < width; i++) s += qkv[r * stride + h * width + i] * qkv[n * stride + dim + h * width + i];
      scores.push(s / Math.sqrt(width));
    }
    const top = Math.max(...scores), weights = scores.map((s) => Math.exp(s - top));
    const sum = weights.reduce((a, w) => a + w, 0);
    for (let i = 0; i < width; i++) {
      let total = 0;
      for (let n = 0; n < keys; n++) total += weights[n] * qkv[n * stride + 2 * dim + h * width + i];
      want[r * yStride + h * width + i] = total / sum;
    }
  }
  const yBuffer = upload(device, y0);
  const step = kernels.attention(upload(device, qkv), yBuffer, { rows, stride, dim, heads, prefix, yStride });
  device.queue.submit([encode(device, step)]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

async function poolKernel(device, kernels, { side, pool, dim, yRow0 }) {
  const rand = random(side * pool + dim), xStride = dim + 4, yStride = dim + 8;
  const cells = Math.ceil(side / pool), x = floats(rand, side * side * xStride);
  const y0 = floats(rand, (yRow0 + cells * cells) * yStride), want = y0.slice();
  for (let cy = 0; cy < cells; cy++) for (let cx = 0; cx < cells; cx++) for (let i = 0; i < dim; i++) {
    let sum = 0, count = 0;
    for (let ry = cy * pool; ry < Math.min(side, (cy + 1) * pool); ry++) {
      for (let rx = cx * pool; rx < Math.min(side, (cx + 1) * pool); rx++) { sum += x[(ry * side + rx) * xStride + i]; count++; }
    }
    want[(yRow0 + cy * cells + cx) * yStride + i] = sum / count;
  }
  const yBuffer = upload(device, y0);
  const step = kernels.poolGrid(upload(device, x), yBuffer, { side, pool, xStride, dim, yRow0, yStride });
  device.queue.submit([encode(device, [step])]);
  return worst(await read(device, yBuffer, 0, y0.byteLength), want);
}

async function argmaxKernel(device, kernels, { rows, segments, width, y0 }) {
  const rand = random(rows * segments + width), xStride = segments * width + 6;
  const x = floats(rand, rows * xStride, 6), want = new Float32Array((y0 + rows * segments) * 2);
  for (let r = 0; r < rows; r++) for (let s = 0; s < segments; s++) {
    const run = x.subarray(r * xStride + s * width, r * xStride + (s + 1) * width);
    let best = 0;
    run.forEach((v, i) => { if (v > run[best]) best = i; });
    want[(y0 + r * segments + s) * 2] = best;
    want[(y0 + r * segments + s) * 2 + 1] = 1 / run.reduce((a, v) => a + Math.exp(v - run[best]), 0);
  }
  const yBuffer = storage(device, want.byteLength);
  device.queue.submit([encode(device, [kernels.argmax(upload(device, x), yBuffer, { rows, xStride, segments, width, y0 })])]);
  return worst(await read(device, yBuffer, 0, want.byteLength), want);
}

export async function run(query, log) {
  const adapter = await navigator.gpu.requestAdapter();
  const f16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice({ requiredFeatures: f16 ? ["shader-f16"] : [] });
  const kernels = new Kernels(device);
  const base = { add: false, activation: false, xRow0: 0, yRow0: 0, yCol0: 0 };
  const cases = [
    ["quantised linear, 4 bits", () => quantised(device, kernels, 4, { ...base, rows: 9, columns: 96, outs: 21 })],
    ["quantised linear, 4 bits, short last group", () => quantised(device, kernels, 4, { ...base, rows: 5, columns: 76, outs: 10 })],
    ["quantised linear, 4 bits, add into offset rows and columns", () => quantised(device, kernels, 4,
      { rows: 6, columns: 64, outs: 17, add: true, activation: false, xRow0: 3, yRow0: 2, yCol0: 4 })],
    ["quantised linear, 4 bits, GELU", () => quantised(device, kernels, 4, { ...base, rows: 4, columns: 64, outs: 9, activation: true })],
    ["quantised linear, 8 bits", () => quantised(device, kernels, 8, { ...base, rows: 7, columns: 76, outs: 13 })],
    ["half linear", () => half(device, kernels, { rows: 7, ins: 64, outs: 11, add: false, xRow0: 0, yRow0: 0 })],
    ["half linear, add into offset rows", () => half(device, kernels, { rows: 6, ins: 10, outs: 70, add: true, xRow0: 2, yRow0: 3 })],
    ["layer norm", () => norm(device, kernels, { rows: 70, dim: 96, yCol0: 0 })],
    ["layer norm into offset columns", () => norm(device, kernels, { rows: 3, dim: 52, yCol0: 8 })],
    ["GELU in place", () => geluKernel(device, kernels, { first: 5, count: 40000 })],
    ["rotary", () => rotaryKernel(device, kernels, { rows: 11, stride: 3 * 128, dim: 128, heads: 2, position0: 3 })],
    ["attention, head width 64, prefix", () => attentionKernel(device, kernels, { rows: 13, dim: 128, heads: 2, prefix: 8 })],
    ["attention, head width 72, all rows", () => attentionKernel(device, kernels, { rows: 10, dim: 144, heads: 2, prefix: 0 })],
    ["pool grid, 2 x 2 cells of a 7 x 7 grid", () => poolKernel(device, kernels, { side: 7, pool: 2, dim: 24, yRow0: 1 })],
    ["pool grid, single cells", () => poolKernel(device, kernels, { side: 5, pool: 1, dim: 16, yRow0: 0 })],
    ["argmax and probability", () => argmaxKernel(device, kernels, { rows: 6, segments: 2, width: 300, y0: 3 })],
  ];
  // Every tile and query count the runtime may choose on a device.
  for (const [outs, rows] of CHOICES.tiles) {
    const tiled = new Kernels(device, { ...TUNING, outs, rows });
    cases.push([`quantised linear, 4 bits, tile ${outs} x ${rows}`, () => quantised(device, tiled, 4,
      { rows: 21, columns: 76, outs: 19, add: true, activation: true, xRow0: 1, yRow0: 2, yCol0: 4 })]);
  }
  // Products in f16 where the GPU has it: checked to f16's precision.
  for (const [outs, rows] of f16 ? CHOICES.tiles : []) {
    const tiled = new Kernels(device, { ...TUNING, outs, rows, half: true });
    cases.push([`quantised linear, 4 bits, tile ${outs} x ${rows}, f16`, () => quantised(device, tiled, 4,
      { rows: 21, columns: 76, outs: 19, add: true, activation: true, xRow0: 1, yRow0: 2, yCol0: 4 }), 4e-3]);
  }
  for (const queries of CHOICES.queries) {
    const grouped = new Kernels(device, { ...TUNING, queries });
    cases.push([`attention, ${queries} query rows per invocation`,
      () => attentionKernel(device, grouped, { rows: 13, dim: 128, heads: 2, prefix: 7 })]);
  }
  const split = new Kernels(device, { ...TUNING, split: true });
  cases.push(["attention in three kernels, head width 64, prefix",
    () => attentionKernel(device, split, { rows: 13, dim: 128, heads: 2, prefix: 7 })]);
  cases.push(["attention in three kernels, head width 72, all rows",
    () => attentionKernel(device, split, { rows: 10, dim: 144, heads: 3, prefix: 0 })]);
  cases.push(["attention in three kernels, more rows than a tile",
    () => attentionKernel(device, split, { rows: 37, dim: 128, heads: 2, prefix: 21 })]);
  const results = { passed: 0, failed: 0, cases: [] };
  for (const [name, test, tolerance = 2e-5] of cases) {
    const error = await test();
    const ok = error < tolerance;
    results[ok ? "passed" : "failed"]++;
    results.cases.push({ name, error, ok });
    log(`${ok ? "ok  " : "FAIL"} ${name}: largest error ${error.toExponential(2)} of the largest value`);
  }
  return results;
}
