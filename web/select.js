// From the model's per-slot outputs to reported boxes: decoding bins as Moondream does,
// scoring, and the selection of `src/crescent_moon/merge.py` for one full-image view
// (threshold on the score, then weighted box fusion of same-label duplicates).

const BINS = 1024;

// Slots as candidates: corners in [0, 1] of the crop, label (null when the tokens spell
// no category), score = objectness x label confidence x box confidence ^ boxPower.
export function candidates(slots, vocabulary, boxPower) {
  const names = new Map(Object.entries(vocabulary).map(([name, ids]) => [ids.join(","), name]));
  return slots.map((slot) => {
    const [bx, by, bw, bh] = slot.bins;
    const cx = bx / BINS, cy = by / BINS;
    const w = 2 ** (bw / (BINS - 1) * 10 - 10), h = 2 ** (bh / (BINS - 1) * 10 - 10);
    const clamp = (v) => Math.min(1, Math.max(0, v));
    const [px, py, pw, ph] = slot.binProbabilities;
    const boxConfidence = Math.min(px, py, Math.min(pw, ph));
    const labelConfidence = slot.tokenProbabilities.reduce((a, p) => a * p, 1);
    const objectness = 1 / (1 + Math.exp(-slot.objectness));
    return {
      corners: [clamp(cx - w / 2), clamp(cy - h / 2), clamp(cx + w / 2), clamp(cy + h / 2)],
      label: names.get(slot.tokens.join(",")) ?? null,
      score: objectness * labelConfidence * boxConfidence ** boxPower,
      objectness, boxConfidence, labelConfidence,
    };
  });
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const area = (r) => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
  const union = area(a) + area(b) - w * h;
  return union > 0 ? (w * h) / union : 0;
}

// Weighted box fusion: in descending score, a box joins the group whose fused box it
// overlaps most if that overlap reaches `duplicateIou`; a group reports the
// score-weighted mean of its boxes with its best score.
function fuse(boxes, duplicateIou) {
  const groups = [];
  for (const c of [...boxes].sort((a, b) => b.score - a.score)) {
    let best = -1, overlap = -1;
    groups.forEach((g, i) => {
      const v = iou(c.box, g.box);
      if (v > overlap) { overlap = v; best = i; }
    });
    if (best >= 0 && overlap >= duplicateIou) {
      const g = groups[best];
      g.members.push(c);
      const total = g.members.reduce((sum, m) => sum + m.score, 0);
      g.box = [0, 1, 2, 3].map((i) => g.members.reduce((sum, m) => sum + m.score * m.box[i], 0) / total);
    } else {
      groups.push({ box: c.box.slice(), score: c.score, members: [c] });
    }
  }
  return groups.map((g) => ({ box: g.box, score: g.score }));
}

// Boxes [{ label, box [x0, y0, x1, y1] in pixels, score }] of the requested labels, per
// label in descending score. `labels` null means every label of the vocabulary.
export function select(found, width, height, { threshold, duplicateIou, labels = null }) {
  const byLabel = new Map();
  for (const c of found) {
    if (c.label === null || c.score < threshold || (labels && !labels.includes(c.label))) continue;
    const [x0, y0, x1, y1] = c.corners;
    const box = [x0 * width, y0 * height, x1 * width, y1 * height];
    if (!byLabel.has(c.label)) byLabel.set(c.label, []);
    byLabel.get(c.label).push({ box, score: c.score });
  }
  const out = [];
  for (const [label, boxes] of byLabel) {
    for (const kept of fuse(boxes, duplicateIou).sort((a, b) => b.score - a.score)) out.push({ label, ...kept });
  }
  return out;
}
