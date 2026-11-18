// Seconds per frame of YOLO models in this browser, as a reference for `bench`: ONNX
// files (`<name>.onnx`, 640 x 640 input, next to the export) run through ONNX Runtime
// Web's WebGPU backend, loaded from a CDN. The time covers the network's pass and reading
// its output back; box decoding and non-maximum suppression are not included.

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const SIDE = 640;

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export async function run(query, log) {
  const ort = await import(`${ORT}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT;
  const base = new URL(query.get("export"), location.href);
  const frames = Number(query.get("frames") || 6);
  const pixels = new Float32Array(3 * SIDE * SIDE).map(() => Math.random());
  const input = new ort.Tensor("float32", pixels, [1, 3, SIDE, SIDE]);
  const models = {};
  for (const name of (query.get("models") || "yolo11n,yolo11x").split(",")) {
    const started = performance.now();
    const session = await ort.InferenceSession.create(new URL(`${name}.onnx`, base).href,
      { executionProviders: ["webgpu"] });
    const load = (performance.now() - started) / 1000;
    const seconds = [];
    for (let i = 0; i < frames; i++) {
      const begun = performance.now();
      const outputs = await session.run({ [session.inputNames[0]]: input });
      await outputs[session.outputNames[0]].getData();
      seconds.push((performance.now() - begun) / 1000);
    }
    // The first frame compiles the shaders; the median is over the rest.
    models[name] = { load_seconds: load, frames: seconds, median: median(seconds.slice(1)) };
    log(`${name}: ${JSON.stringify(models[name])}`);
    await session.release();
  }
  return { passed: 1, failed: 0, models };
}
