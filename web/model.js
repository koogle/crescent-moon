// The slot detector's forward pass on WebGPU: loading the export of
// `scripts/export_web.py` and running one 378 x 378 crop through the vision encoder, the
// projection, the decoder (image prefix and slots together, block by block) and the
// read-out heads. `docs/webgpu.md` describes the pass.

import { Kernels, encode, storage, upload } from "./kernels.js";

const PATCH = 14, GRID = 27, CROP = PATCH * GRID;
const IMAGE_ROWS = GRID * GRID; // 729 image positions
const VISION = { dim: 1152, hidden: 4304, heads: 16, blocks: 27 };
const TEXT = { dim: 2048, hidden: 8192, heads: 32, blocks: 24, vocabulary: 51200 };
const BINS = 1024;
const pad32 = (n) => Math.ceil(n / 32) * 32;

function halfToFloat(bytes, count) {
  const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const h = halves[i], exponent = (h >> 10) & 31, fraction = h & 1023;
    const magnitude = exponent === 0 ? fraction * 2 ** -24
      : exponent === 31 ? (fraction ? NaN : Infinity)
      : (fraction + 1024) * 2 ** (exponent - 25);
    out[i] = h & 0x8000 ? -magnitude : magnitude;
  }
  return out;
}

// Fetches a shard, from the browser's cache when it is there. `version` identifies the
// shard's content: a cached copy of the same file with another version is replaced.
async function fetchShard(url, version, onBytes) {
  const cache = "caches" in globalThis ? await caches.open("crescent-moon-weights") : null;
  const key = `${url}?v=${version}`;
  let response = cache && await cache.match(key);
  if (!response) {
    if (cache) {
      for (const request of await cache.keys()) {
        if (request.url.split("?")[0] === url) await cache.delete(request);
      }
    }
    response = await fetch(key);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    if (cache) {
      try { await cache.put(key, response.clone()); } catch { /* over quota: keep going uncached */ }
    }
  }
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.byteLength;
    onBytes(value.byteLength);
  }
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.byteLength; }
  return bytes;
}

// Reads every tensor of an export into GPU buffers. Quantised layers become
// { weight, scale, bias, rows, columns, groups, scaleStride, bits }; dense FP16 tensors
// named in `packedHalf` stay packed (the half-linear kernel decodes them); the others
// become f32 buffers. `progress(loadedBytes, totalBytes)` is called as shards arrive.
export async function loadWeights(device, base, manifest, progress = () => {}) {
  const places = [];
  for (const [name, entry] of Object.entries(manifest.tensors)) {
    const parts = entry.format === "quantised" ? ["weight", "scale", "bias"] : ["data"];
    for (const part of parts) places.push({ name, part, entry, place: entry[part] });
  }
  const total = places.reduce((sum, p) => sum + p.place.bytes, 0);
  let loaded = 0;
  const tensors = {};
  for (let shard = 0; shard < manifest.shards.length; shard++) {
    const version = places.find((p) => p.place.shard === shard).place.sha256.slice(0, 16);
    const bytes = await fetchShard(new URL(manifest.shards[shard], base).href, version, (n) => {
      loaded += n;
      progress(loaded, total);
    });
    for (const { name, part, entry, place } of places) {
      if (place.shard !== shard) continue;
      const blob = bytes.subarray(place.offset, place.offset + place.bytes);
      const tensor = (tensors[name] ??= {});
      if (entry.format === "quantised") {
        const groups = Math.ceil(entry.columns / manifest.group);
        Object.assign(tensor, {
          rows: entry.rows, columns: entry.columns, bits: entry.bits, groups,
          scaleStride: groups + (groups % 2),
        });
        tensor[part] = part === "bias" ? upload(device, halfToFloat(blob, entry.rows)) : upload(device, blob);
      } else {
        const columns = entry.shape[entry.shape.length - 1];
        const count = place.bytes / 2;
        tensor.shape = entry.shape;
        const dense = /^slots\.(adapters\..*|objectness\.weight)$/.test(name);
        if (dense && columns % 2) throw new Error(`${name}: odd row length`);
        tensor.buffer = dense ? upload(device, blob) : upload(device, halfToFloat(blob, count));
        if (name === "slots.objectness.bias") tensor.value = halfToFloat(blob, 1)[0];
      }
    }
  }
  return tensors;
}

// Rotary angles as Moondream's precompute_freqs_cis(32, positions): (cos, sin) per
// position and each of 16 frequencies.
function rotaryTable(positions) {
  const table = new Float32Array(positions * 32);
  for (let position = 0; position < positions; position++) {
    for (let j = 0; j < 16; j++) {
      const frequency = Math.fround(1 / Math.fround(Math.pow(10000, Math.fround(2 * j) / 32)));
      const angle = Math.fround(position * frequency);
      table[position * 32 + 2 * j] = Math.cos(angle);
      table[position * 32 + 2 * j + 1] = Math.sin(angle);
    }
  }
  return table;
}

// Patch rows [729, 608] of a 378 x 378 RGBA or RGB crop: per patch, channel by channel,
// row by row, pixels scaled to [-1, 1]; the last 20 columns of each row stay zero.
export function patches(pixels, channels) {
  const stride = pad32(3 * PATCH * PATCH);
  const out = new Float32Array(IMAGE_ROWS * stride);
  for (let py = 0; py < GRID; py++) {
    for (let px = 0; px < GRID; px++) {
      const row = (py * GRID + px) * stride;
      for (let c = 0; c < 3; c++) {
        for (let y = 0; y < PATCH; y++) {
          const source = ((py * PATCH + y) * CROP + px * PATCH) * channels + c;
          const target = row + (c * PATCH + y) * PATCH;
          for (let x = 0; x < PATCH; x++) out[target + x] = pixels[source + x * channels] / 127.5 - 1;
        }
      }
    }
  }
  return out;
}

export class SlotModel {
  // `tensors` from loadWeights; `manifest` of the same export.
  constructor(device, manifest, tensors) {
    this.device = device;
    this.manifest = manifest;
    this.slots = manifest.head.slots;
    this.width = manifest.selection.coordinates + manifest.head.label_tokens;
    this.slotRows = this.slots * this.width;
    // The image prefix: BOS, then the image positions, pooled when the head was trained so.
    this.pool = manifest.head.prefix_pool;
    this.prefix = 1 + Math.ceil(GRID / this.pool) ** 2;
    this.rows = this.prefix + this.slotRows;
    this.objectnessBias = tensors["slots.objectness.bias"].value;
    this.kernels = new Kernels(device);
    this.build(tensors);
  }

  build(t) {
    const k = this.kernels, device = this.device;
    const f32 = (count) => storage(device, count * 4);
    const V = VISION, T = TEXT, rows = this.rows, slotRows = this.slotRows, PREFIX = this.prefix;
    const labelTokens = this.manifest.head.label_tokens, coordinates = this.manifest.selection.coordinates;
    const hiddenStride = pad32(V.hidden);
    const b = this.buffers = {
      patches: f32(IMAGE_ROWS * pad32(3 * PATCH * PATCH)),
      vx: f32(IMAGE_ROWS * V.dim), vn: f32(IMAGE_ROWS * V.dim), vqkv: f32(IMAGE_ROWS * 3 * V.dim),
      vmix: f32(IMAGE_ROWS * V.dim), vh: f32(IMAGE_ROWS * hiddenStride),
      pin: f32(IMAGE_ROWS * 2 * V.dim), ph: f32(IMAGE_ROWS * T.hidden),
      projected: f32(IMAGE_ROWS * T.dim),
      x: f32(rows * T.dim), normed: f32(rows * T.dim), qkv: f32(rows * 3 * T.dim),
      mixed: f32(rows * T.dim), hidden: f32(rows * T.hidden), down: f32(slotRows * this.manifest.head.rank),
      coordIn: f32(this.slots * 2 * T.dim), sizeIn: f32(this.slots * T.dim),
      labelIn: f32(this.slots * labelTokens * T.dim), objectIn: f32(this.slots * T.dim),
      readHidden: f32(this.slots * labelTokens * T.hidden),
      coordOut: f32(this.slots * 2 * BINS), sizeOut: f32(this.slots * 2 * BINS),
      labelOut: f32(this.slots * labelTokens * T.vocabulary),
      // objectness [slots], then (index, probability) pairs: x and y bins [slots, 2],
      // width and height bins [slots, 2], label tokens [slots, label tokens]
      results: f32(this.slots * (1 + 2 * (4 + labelTokens))),
    };
    // The decoder's input: BOS, room for the image embeddings, then every slot position.
    b.initial = f32(rows * T.dim);
    b.rotary = upload(device, rotaryTable(rows));
    this.setup = [
      { copy: [t["text.bos"].buffer, 0, b.initial, 0, T.dim * 4] },
      { copy: [t["slots.inputs"].buffer, 0, b.initial, PREFIX * T.dim * 4, slotRows * T.dim * 4] },
    ];
    device.queue.submit([encode(device, this.setup)]);

    const stages = this.stages = { vision: [], projection: [], decoder: [], readout: [] };
    const q = (x, name, y, v) => k.quantisedLinear(t[name].bits, x, t[name], y, v);
    const norm = (x, name, y, v) => k.layerNorm(x, t[`${name}.weight`].buffer, t[`${name}.bias`].buffer, y, v);

    // Vision encoder: patch embedding added to the position embedding, 27 pre-norm blocks.
    let s = stages.vision;
    s.push({ copy: [t["vision.pos_emb"].buffer, 0, b.vx, 0, IMAGE_ROWS * V.dim * 4] });
    s.push(q(b.patches, "vision.patch_emb", b.vx,
      { rows: IMAGE_ROWS, xStride: pad32(3 * PATCH * PATCH), yStride: V.dim, add: true }));
    for (let i = 0; i < V.blocks; i++) {
      const name = `vision.blocks.${i}`, all = { rows: IMAGE_ROWS, xStride: V.dim, yStride: V.dim };
      s.push(norm(b.vx, `${name}.ln1`, b.vn, { ...all, dim: V.dim }));
      s.push(q(b.vn, `${name}.attn.qkv`, b.vqkv, { ...all, yStride: 3 * V.dim }));
      s.push(k.attention(b.vqkv, b.vmix,
        { rows: IMAGE_ROWS, stride: 3 * V.dim, dim: V.dim, heads: V.heads, yStride: V.dim }));
      s.push(q(b.vmix, `${name}.attn.proj`, b.vx, { ...all, add: true }));
      s.push(norm(b.vx, `${name}.ln2`, b.vn, { ...all, dim: V.dim }));
      s.push(q(b.vn, `${name}.mlp.fc1`, b.vh, { ...all, yStride: hiddenStride, gelu: true }));
      s.push(q(b.vh, `${name}.mlp.fc2`, b.vx, { ...all, xStride: hiddenStride, add: true }));
    }

    // Projection: with one crop the local features are the global ones, side by side.
    s = stages.projection;
    for (const yCol0 of [0, V.dim]) {
      s.push(norm(b.vx, "vision.post_ln", b.pin,
        { rows: IMAGE_ROWS, xStride: V.dim, dim: V.dim, yStride: 2 * V.dim, yCol0 }));
    }
    s.push(q(b.pin, "vision.proj_mlp.fc1", b.ph,
      { rows: IMAGE_ROWS, xStride: 2 * V.dim, yStride: T.hidden, gelu: true }));
    s.push({ copy: [b.initial, 0, b.x, 0, rows * T.dim * 4] });
    s.push(q(b.ph, "vision.proj_mlp.fc2", b.projected,
      { rows: IMAGE_ROWS, xStride: T.hidden, yStride: T.dim }));
    s.push(k.poolGrid(b.projected, b.x,
      { side: GRID, pool: this.pool, xStride: T.dim, dim: T.dim, yRow0: 1, yStride: T.dim }));

    // Decoder: prefix rows attend to the prefix, slot rows to everything; adapters add
    // to slot rows only.
    s = stages.decoder;
    const rank = this.manifest.head.rank;
    const adapt = (block, part, x, xStride, ins, y, yStride, outs) => {
      const name = `slots.adapters.${block}.${part}`;
      s.push(k.halfLinear(x, t[`${name}.down`].buffer, b.down,
        { rows: slotRows, xRow0: PREFIX, xStride, ins, outs: rank, yStride: rank }));
      s.push(k.halfLinear(b.down, t[`${name}.up`].buffer, y,
        { rows: slotRows, xStride: rank, ins: rank, outs, yRow0: PREFIX, yStride, add: true }));
    };
    this.blockEnds = [];
    for (let i = 0; i < T.blocks; i++) {
      const name = `text.blocks.${i}`, all = { rows, xStride: T.dim, yStride: T.dim };
      s.push(norm(b.x, `${name}.ln`, b.normed, { ...all, dim: T.dim }));
      s.push(q(b.normed, `${name}.attn.qkv`, b.qkv, { ...all, yStride: 3 * T.dim }));
      adapt(i, "qkv", b.normed, T.dim, T.dim, b.qkv, 3 * T.dim, 3 * T.dim);
      s.push(k.rotary(b.rotary, b.qkv, { rows, stride: 3 * T.dim, dim: T.dim, heads: T.heads }));
      s.push(k.attention(b.qkv, b.mixed,
        { rows, stride: 3 * T.dim, dim: T.dim, heads: T.heads, prefix: PREFIX, yStride: T.dim }));
      s.push(q(b.mixed, `${name}.attn.proj`, b.x, { ...all, add: true }));
      adapt(i, "proj", b.mixed, T.dim, T.dim, b.x, T.dim, T.dim);
      s.push(q(b.normed, `${name}.mlp.fc1`, b.hidden, { ...all, yStride: T.hidden }));
      adapt(i, "fc1", b.normed, T.dim, T.dim, b.hidden, T.hidden, T.hidden);
      s.push(k.gelu(b.hidden, 0, rows * T.hidden));
      s.push(q(b.hidden, `${name}.mlp.fc2`, b.x, { ...all, xStride: T.hidden, add: true }));
      adapt(i, "fc2", b.hidden, T.hidden, T.hidden, b.x, T.dim, T.dim);
      this.blockEnds.push(s.length);
    }

    // Read-out. Positions 0 and 1 of a slot go to the coordinate decoder and position 2
    // to the size decoder, from the raw hidden state; the label positions and the
    // objectness layer read the normalised one.
    s = stages.readout;
    const row = T.dim * 4;
    s.push(norm(b.x, "text.post_ln", b.normed,
      { rows: slotRows, xRow0: PREFIX, xStride: T.dim, dim: T.dim, yRow0: PREFIX, yStride: T.dim }));
    for (let slot = 0; slot < this.slots; slot++) {
      const from = (PREFIX + slot * this.width) * row;
      s.push({ copy: [b.x, from, b.coordIn, slot * 2 * row, 2 * row] });
      s.push({ copy: [b.x, from + 2 * row, b.sizeIn, slot * row, row] });
      s.push({ copy: [b.normed, from, b.objectIn, slot * row, row] });
      s.push({ copy: [b.normed, from + coordinates * row, b.labelIn, slot * labelTokens * row, labelTokens * row] });
    }
    const pairs = this.slots * 2; // f32 offset of (index, probability) pairs after objectness
    s.push(k.halfLinear(b.objectIn, t["slots.objectness.weight"].buffer, b.results,
      { rows: this.slots, xStride: T.dim, ins: T.dim, outs: 1, yStride: 1 }));
    s.push(q(b.coordIn, "region.coord_decoder.fc1", b.readHidden,
      { rows: this.slots * 2, xStride: T.dim, yStride: T.hidden, gelu: true }));
    s.push(q(b.readHidden, "region.coord_decoder.fc2", b.coordOut,
      { rows: this.slots * 2, xStride: T.hidden, yStride: BINS }));
    s.push(k.argmax(b.coordOut, b.results,
      { rows: this.slots * 2, xStride: BINS, segments: 1, width: BINS, y0: this.slots / 2 }));
    s.push(q(b.sizeIn, "region.size_decoder.fc1", b.readHidden,
      { rows: this.slots, xStride: T.dim, yStride: T.hidden, gelu: true }));
    s.push(q(b.readHidden, "region.size_decoder.fc2", b.sizeOut,
      { rows: this.slots, xStride: T.hidden, yStride: 2 * BINS }));
    s.push(k.argmax(b.sizeOut, b.results,
      { rows: this.slots, xStride: 2 * BINS, segments: 2, width: BINS, y0: this.slots / 2 + pairs }));
    s.push(q(b.labelIn, "text.lm_head", b.labelOut,
      { rows: this.slots * labelTokens, xStride: T.dim, yStride: T.vocabulary }));
    s.push(k.argmax(b.labelOut, b.results, {
      rows: this.slots * labelTokens, xStride: T.vocabulary, segments: 1, width: T.vocabulary,
      y0: this.slots / 2 + 2 * pairs,
    }));
    this.resultBytes = b.results.size;
    this.staging = device.createBuffer({
      size: this.resultBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  // Runs one crop. `patchRows` from patches(). Returns per slot the objectness logit,
  // bins [x, y, w, h] with their probabilities and label tokens with theirs, and the
  // seconds each stage took (the GPU queue is drained after each stage when `timed`).
  // `probe(stage, index, total)` may be async and is awaited after every step when given;
  // tests use it to read intermediate buffers.
  async run(patchRows, { timed = false, probe = null } = {}) {
    const device = this.device, b = this.buffers;
    device.queue.writeBuffer(b.patches, 0, patchRows);
    const seconds = {};
    for (const [stage, steps] of Object.entries(this.stages)) {
      const start = performance.now();
      if (probe) {
        for (let i = 0; i < steps.length; i++) {
          device.queue.submit([encode(device, [steps[i]])]);
          await probe(stage, i + 1, steps.length);
        }
      } else {
        // One submission per decoder block keeps command buffers short.
        const cuts = stage === "decoder" ? this.blockEnds : [steps.length];
        let from = 0;
        for (const to of cuts) {
          device.queue.submit([encode(device, steps.slice(from, to))]);
          from = to;
        }
      }
      if (timed) {
        await device.queue.onSubmittedWorkDone();
        seconds[stage] = (performance.now() - start) / 1000;
      }
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(b.results, 0, this.staging, 0, this.resultBytes);
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    const n = this.slots, labelTokens = this.manifest.head.label_tokens;
    const slots = [];
    for (let slot = 0; slot < n; slot++) {
      const pair = (base, i) => [out[n + 2 * (base + i)], out[n + 2 * (base + i) + 1]];
      const picks = [pair(0, slot * 2), pair(0, slot * 2 + 1), pair(2 * n, slot * 2), pair(2 * n, slot * 2 + 1)];
      const tokens = Array.from({ length: labelTokens }, (_, i) => pair(4 * n, slot * labelTokens + i));
      slots.push({
        objectness: out[slot] + this.objectnessBias,
        bins: picks.map((p) => p[0]), binProbabilities: picks.map((p) => p[1]),
        tokens: tokens.map((p) => p[0]), tokenProbabilities: tokens.map((p) => p[1]),
      });
    }
    return { slots, seconds };
  }
}

// Requests a device for the model: timestamp-free, with the storage limits it needs.
// `features` are requested where the adapter has them.
export async function requestDevice(features = []) {
  if (!navigator.gpu) throw new Error("This browser has no WebGPU");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  const need = 256 * 2 ** 20;
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(need, adapter.limits.maxStorageBufferBindingSize),
    maxBufferSize: Math.min(need, adapter.limits.maxBufferSize),
  };
  const requiredFeatures = features.filter((feature) => adapter.features.has(feature));
  return adapter.requestDevice({ requiredLimits, requiredFeatures });
}

export async function loadModel(base, progress) {
  const device = await requestDevice();
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const tensors = await loadWeights(device, base, manifest, progress);
  return new SlotModel(device, manifest, tensors);
}

export { CROP };
