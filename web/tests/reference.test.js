// The full pass on an export of scripts/export_web.py against the PyTorch reference
// activations stored with it: vision features, projected embeddings, the prefix hidden
// state after decoder blocks 0, 11 and 23, the slots' final hidden state, every slot's
// objectness, bins, tokens and probabilities, and the selected boxes.

import { read } from "../kernels.js";
import { SlotModel, loadWeights, patches, requestDevice } from "../model.js";
import { candidates, select } from "../select.js";

const TEXT_DIM = 2048, VISION_DIM = 1152, IMAGE_ROWS = 729;
// Largest difference allowed, as a share of the reference's largest magnitude.
const ACTIVATION_TOLERANCE = 2e-3;

async function array(base, entry) {
  const bytes = await (await fetch(new URL(entry.file, base))).arrayBuffer();
  const kinds = { float32: Float32Array, int32: Int32Array, uint8: Uint8Array };
  return new kinds[entry.dtype](bytes);
}

function difference(got, want) {
  let error = 0, size = 0;
  for (let i = 0; i < want.length; i++) {
    error = Math.max(error, Math.abs(got[i] - want[i]));
    size = Math.max(size, Math.abs(want[i]));
  }
  return error / size;
}

function columns(matrix, rows, stride, width) {
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r++) out.set(matrix.subarray(r * stride, r * stride + width), r * width);
  return out;
}

export async function run(query, log) {
  const base = new URL(query.get("export"), location.href);
  const device = await requestDevice();
  device.addEventListener("uncapturederror", (event) => log(`GPU ERROR ${event.error.message}`));
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const started = performance.now();
  const tensors = await loadWeights(device, base, manifest);
  const model = new SlotModel(device, manifest, tensors);
  log(`loaded ${manifest.shards.length} shards in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  const results = { passed: 0, failed: 0, images: {} };
  const check = (image, name, ok, detail) => {
    results[ok ? "passed" : "failed"]++;
    (results.images[image] ??= []).push({ name, ok, ...detail });
    log(`${ok ? "ok  " : "FAIL"} ${image} ${name}: ${JSON.stringify(detail)}`);
  };
  const images = query.get("image") ? [query.get("image")] : manifest.reference;
  for (const image of images) {
    const folder = new URL(`reference/${image}/`, base);
    const reference = await (await fetch(new URL("reference.json", folder))).json();
    const want = {};
    for (const [name, entry] of Object.entries(reference.arrays)) want[name] = await array(folder, entry);
    const got = {};
    const b = model.buffers, row = TEXT_DIM * 4;
    const blocks = { 0: "prefix_00", 11: "prefix_11", 23: "prefix_23" };
    const begun = performance.now();
    const probe = async (stage, index, total) => {
      if (index === total) {
        await device.queue.onSubmittedWorkDone();
        log(`  ${image}: ${stage} done ${((performance.now() - begun) / 1000).toFixed(0)} s after the start`);
      }
      if (stage === "projection" && index === total) {
        const pin = await read(device, b.pin, 0, IMAGE_ROWS * 2 * VISION_DIM * 4);
        got.vision = columns(pin, IMAGE_ROWS, 2 * VISION_DIM, VISION_DIM);
        got.projection = await read(device, b.projected, 0, IMAGE_ROWS * row);
      }
      const block = stage === "decoder" ? model.blockEnds.indexOf(index) : -1;
      if (block in blocks) got[blocks[block]] = await read(device, b.x, 0, model.prefix * row);
      if (block === 23) got.slots = await read(device, b.x, model.prefix * row, model.slotRows * row);
      if (stage === "decoder" && block >= 0) log(`  ${image}: decoder block ${block} done`);
    };
    const start = performance.now();
    const { slots } = await model.run(patches(want.crop, 3), { probe });
    const seconds = (performance.now() - start) / 1000;
    for (const name of ["vision", "projection", "prefix_00", "prefix_11", "prefix_23", "slots"]) {
      const error = difference(got[name], want[name]);
      check(image, name, error < ACTIVATION_TOLERANCE, { error });
    }
    const flat = (key) => slots.flatMap((s) => s[key]);
    const same = (a, w) => a.reduce((n, v, i) => n + (v === w[i]), 0);
    const objectness = difference(slots.map((s) => s.objectness), want.objectness);
    check(image, "objectness", objectness < ACTIVATION_TOLERANCE, { error: objectness });
    const bins = same(flat("bins"), want.bins), tokens = same(flat("tokens"), want.tokens);
    check(image, "bins", bins === want.bins.length, { same: bins, of: want.bins.length });
    check(image, "tokens", tokens === want.tokens.length, { same: tokens, of: want.tokens.length });
    const binProbability = difference(flat("binProbabilities"), want.bin_probability);
    const tokenProbability = difference(flat("tokenProbabilities"), want.token_probability);
    check(image, "bin probabilities", binProbability < 1e-2, { error: binProbability });
    check(image, "token probabilities", tokenProbability < 1e-2, { error: tokenProbability });
    const found = select(candidates(slots, manifest.vocabulary, manifest.selection.box_power),
      reference.width, reference.height,
      { threshold: manifest.selection.threshold, duplicateIou: manifest.selection.duplicate_iou });
    // The reference lists boxes per label in vocabulary order; compare as sets.
    const key = (box) => `${box.label} ${box.box.map((v) => v.toFixed(1)).join(" ")} ${box.score.toFixed(3)}`;
    const wanted = reference.boxes.map(key).sort(), reported = found.map(key).sort();
    const boxes = wanted.length === reported.length && wanted.every((v, i) => v === reported[i]);
    check(image, "boxes", boxes, { reference: wanted.length, reported: reported.length });
    if (!boxes) log(`  reference ${JSON.stringify(wanted)}\n  reported  ${JSON.stringify(reported)}`);
    log(`${image}: pass with probes took ${seconds.toFixed(1)} s`);
    results.images[image].seconds = seconds;
  }
  return results;
}
