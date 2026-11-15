// Runs the first steps of the pass on an export and reports each written buffer's
// statistics, for finding the step where the browser departs from PyTorch:
//   node web/tests/run.cjs steps --export artifacts/web/NAME --image STEM
import { read } from "../kernels.js";
import { SlotModel, loadWeights, patches, requestDevice } from "../model.js";

export async function run(query, log) {
  const base = new URL(query.get("export"), location.href);
  const device = await requestDevice();
  device.addEventListener("uncapturederror", (event) => log(`GPU ERROR ${event.error.message}`));
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  const model = new SlotModel(device, manifest, await loadWeights(device, base, manifest));
  const image = query.get("image") || manifest.reference[0];
  const folder = new URL(`reference/${image}/`, base);
  const crop = new Uint8Array(await (await fetch(new URL("crop.u8", folder))).arrayBuffer());
  const b = model.buffers, limit = Number(query.get("steps") || 12);
  const watched = ["vx", "vn", "vqkv", "vmix", "vh"];
  const rows = [];
  class Done extends Error {}
  try {
    await model.run(patches(crop, 3), {
      probe: async (stage, index) => {
        const stats = { stage, index, label: model.stages[stage][index - 1].label ?? "copy" };
        for (const name of watched) {
          const values = await read(device, b[name], 0, b[name].size);
          let nan = 0, top = 0, sum = 0;
          for (const v of values) { if (Number.isNaN(v)) nan++; else { top = Math.max(top, Math.abs(v)); sum += Math.abs(v); } }
          stats[name] = `nan ${nan} max ${top.toPrecision(4)} mean ${(sum / values.length).toPrecision(4)}`;
        }
        rows.push(stats);
        log(JSON.stringify(stats));
        if (index >= limit) throw new Done();
      },
    });
  } catch (error) {
    if (!(error instanceof Done)) throw error;
  }
  return { failed: 0, rows };
}
