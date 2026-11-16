// Seconds per stage of the full pass on an export, over several frames of one reference
// crop, with the adapter the browser chose: the speed measurement of the runtime.
//
//   node web/tests/run.cjs bench --export artifacts/web/NAME [--frames N] [--gpu vulkan]

import { TUNING, profile } from "../kernels.js";
import { SlotModel, loadWeights, patches, requestDevice, tune } from "../model.js";

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export async function run(query, log) {
  const base = new URL(query.get("export"), location.href);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const info = adapter.info ?? {};
  const described = {
    vendor: info.vendor, architecture: info.architecture, device: info.device,
    description: info.description, fallback: adapter.isFallbackAdapter ?? info.isFallbackAdapter,
    features: [...adapter.features].sort(),
  };
  log(`adapter: ${JSON.stringify(described)}`);
  const device = await requestDevice(["timestamp-query", "shader-f16"]);
  device.addEventListener("uncapturederror", (event) => log(`GPU ERROR ${event.error.message}`));
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const started = performance.now();
  const tensors = await loadWeights(device, base, manifest);
  const load = (performance.now() - started) / 1000;
  log(`loaded in ${load.toFixed(1)} s`);
  const tuneStarted = performance.now();
  const limits = {};
  for (const name of ["budget", "slow", "half"]) if (query.get(name)) limits[name] = Number(query.get(name));
  const tuning = query.get("tuning")
    ? { ...TUNING, ...JSON.parse(query.get("tuning")) }
    : await tune(device, manifest, tensors, limits);
  const tuneSeconds = (performance.now() - tuneStarted) / 1000;
  log(`tuned in ${tuneSeconds.toFixed(1)} s: ${JSON.stringify(tuning)}`);
  const model = new SlotModel(device, manifest, tensors, tuning);
  const image = query.get("image") || manifest.reference[0];
  const crop = new Uint8Array(await (await fetch(new URL(`reference/${image}/crop.u8`, base))).arrayBuffer());
  const rows = patches(crop, 3);
  const frames = [];
  for (let i = 0; i < Number(query.get("frames") || 5); i++) {
    const begun = performance.now();
    const { seconds } = await model.run(rows, { timed: true });
    frames.push({ ...seconds, total: (performance.now() - begun) / 1000 });
    log(`frame ${i}: ${JSON.stringify(frames[i])}`);
  }
  // The first frame compiles the pipelines; the median is over the rest.
  const warm = frames.slice(1);
  const medians = Object.fromEntries(Object.keys(frames[0]).map((k) => [k, median(warm.map((f) => f[k]))]));
  log(`median of warm frames: ${JSON.stringify(medians)}`);
  // One more frame with GPU timestamps around every kernel: seconds per stage and kernel.
  const kernels = {};
  if (device.features.has("timestamp-query")) {
    device.queue.writeBuffer(model.buffers.patches, 0, rows);
    for (const [stage, steps] of Object.entries(model.stages)) {
      for (const [label, seconds] of Object.entries(await profile(device, steps))) {
        kernels[`${stage} ${label}`] = seconds;
      }
    }
    log(`GPU seconds per kernel: ${JSON.stringify(kernels)}`);
  }
  return { passed: 1, failed: 0, adapter: described, kernels, export: manifest.head, load_seconds: load, tuning, tune_seconds: tuneSeconds, frames, median: medians };
}
