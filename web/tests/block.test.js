// The first decoder block alone, from the reference's projected embeddings, against
// the reference's prefix hidden state after block 0. A minute on SwiftShader, where the
// full pass of reference.test.js takes half an hour.
//   node web/tests/run.cjs block --export artifacts/web/NAME

import { encode, read } from "../kernels.js";
import { SlotModel, loadWeights, requestDevice } from "../model.js";

export async function run(query, log) {
  const base = new URL(query.get("export"), location.href);
  const device = await requestDevice();
  device.addEventListener("uncapturederror", (event) => log(`GPU ERROR ${event.error.message}`));
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const model = new SlotModel(device, manifest, await loadWeights(device, base, manifest));
  const image = query.get("image") || manifest.reference[0];
  const folder = new URL(`reference/${image}/`, base);
  const f32 = async (name) => new Float32Array(await (await fetch(new URL(name, folder))).arrayBuffer());
  const projection = await f32("projection.f32"), want = await f32("prefix_00.f32");
  const b = model.buffers, row = 2048 * 4;
  device.queue.submit([encode(device, [{ copy: [b.initial, 0, b.x, 0, b.x.size] }])]);
  device.queue.writeBuffer(b.x, row, projection);
  device.queue.submit([encode(device, model.stages.decoder.slice(0, model.blockEnds[0]))]);
  const got = await read(device, b.x, 0, model.prefix * row);
  let error = 0, size = 0;
  for (let i = 0; i < want.length; i++) {
    error = Math.max(error, Math.abs(got[i] - want[i]));
    size = Math.max(size, Math.abs(want[i]));
  }
  const ok = error / size < 1e-3;
  log(`${ok ? "ok  " : "FAIL"} ${image} prefix after block 0: largest error ${(error / size).toExponential(2)} of the largest value`);
  return { passed: +ok, failed: +!ok, error: error / size };
}
