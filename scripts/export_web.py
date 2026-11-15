"""Export a slot head and its backbone for the browser runtime, with reference activations.

    python scripts/export_web.py --head artifacts/heads/RUN/head.pt --out artifacts/web/RUN-w4 \
        --bits 4 --images artifacts/web-images/*.jpg

Writes under `--out`:

- `weights-NNN.bin` shards and `manifest.json`: every Linear the one-pass detector uses,
  rounded to `--bits` bits per group of 32 (`--component-bits heads=8` overrides one of
  vision, projection, text, heads) in the format of `crescent_moon.pack`; biases, norms,
  the position embedding, the BOS embedding, the slot inputs, the adapters and the
  objectness layer in FP16; the label vocabulary and the selection settings.
- `reference/<image>/`: for each image, the 378 x 378 crop the vision encoder reads and
  the activations of the PyTorch pass on exactly the weights the shards hold (float32 on
  the CPU), then every slot and the selected boxes.

The detector runs one crop: with `max_crops` 1 Moondream's local crop is the global crop,
so the vision encoder runs once and its features serve as both.
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from crescent_moon.pack import (
    GROUP,
    Shards,
    pack_half,
    pack_integers,
    unpack_half,
    unpack_integers,
)
from crescent_moon.quantize import dequantise, quantise
from crescent_moon.slots import COORDINATES
from crescent_moon.tasks import COCO_NAMES
from crescent_moon.train_slots import SlotDetector, image_cache, load_backbone

COMPONENTS = {
    "vision": ("vision.patch_emb", "vision.blocks."),
    "projection": ("vision.proj_mlp.",),
    "text": ("text.blocks.",),
    "heads": ("text.lm_head", "region.coord_decoder.", "region.size_decoder."),
}
TRACED_PREFIX_BLOCKS = (0, 11, 23)


def component_of(name):
    for component, prefixes in COMPONENTS.items():
        if name.startswith(prefixes):
            return component
    return None


def half_(tensor):
    """Round a parameter in place to the FP16 values the export stores; returns it."""
    tensor.data = tensor.data.float().half().float()
    return tensor.data


def export_backbone(inner, shards, bits_of):
    """Pack the detector's Linears and small parameters; the model keeps exactly the
    stored values. Returns the manifest's tensor entries."""
    tensors = {}
    for name, layer in inner.named_modules():
        component = component_of(name)
        if not isinstance(layer, torch.nn.Linear) or component is None:
            continue
        bits = bits_of[component]
        integers, scale = quantise(layer.weight.data, bits, GROUP, scale_dtype=torch.float16)
        layer.weight.data = dequantise(integers, scale, GROUP)
        rows, columns = integers.shape
        tensors[name] = {
            "format": "quantised",
            "bits": bits,
            "rows": rows,
            "columns": columns,
            "weight": shards.add(pack_integers(integers.numpy(), bits)),
            "scale": shards.add(pack_half(scale.numpy())),
            "bias": shards.add(pack_half(half_(layer.bias).numpy())),
        }
    for name, parameter in inner.named_parameters():
        if ".ln" in name or "post_ln" in name or name == "vision.pos_emb":
            values = half_(parameter).reshape(-1, parameter.shape[-1]).squeeze(0)
            tensors[name] = {
                "format": "half",
                "shape": list(values.shape),
                "data": shards.add(pack_half(values.numpy())),
            }
    return tensors


def export_head(head, inner, shards):
    """Pack the slot head: its input embeddings, adapters and objectness layer."""
    tensors = {}

    def dense(name, values):
        tensors[name] = {
            "format": "half",
            "shape": list(values.shape),
            "data": shards.add(pack_half(values.detach().numpy())),
        }

    for parameter in head.parameters():
        half_(parameter)
    bos = inner.text.wte[inner.config.tokenizer.bos_id].float().half().float()
    inner.text.wte.data[inner.config.tokenizer.bos_id] = bos
    dense("text.bos", bos)
    # mask + role + slot query, the input of every slot position, in slot order.
    inputs = (head.mask + head.roles)[None] + head.queries[:, None]
    dense("slots.inputs", inputs.flatten(0, 1).half().float())
    for index, adapter in enumerate(head.adapters):
        for part in ("qkv", "proj", "fc1", "fc2"):
            dense(f"slots.adapters.{index}.{part}.down", adapter[part].down.weight.data)
            dense(f"slots.adapters.{index}.{part}.up", adapter[part].up.weight.data)
    dense("slots.objectness.weight", head.objectness.weight.data)
    dense("slots.objectness.bias", head.objectness.bias.data)
    return tensors


def verify(out, manifest, inner, head):
    """Unpack every blob from the written shards and compare with the model bit for bit."""
    shards = [(out / name).read_bytes() for name in manifest["shards"]]

    def blob(place):
        return shards[place["shard"]][place["offset"] : place["offset"] + place["bytes"]]

    modules = dict(inner.named_modules())
    parameters = dict(inner.named_parameters())
    for name, entry in manifest["tensors"].items():
        if entry["format"] == "quantised":
            rows, columns, bits = entry["rows"], entry["columns"], entry["bits"]
            integers = unpack_integers(blob(entry["weight"]), rows, columns, bits)
            scale = unpack_half(blob(entry["scale"]), (rows, -(-columns // GROUP)))
            weight = dequantise(torch.from_numpy(integers.copy()), torch.from_numpy(scale.copy()))
            if not torch.equal(weight, modules[name].weight.data):
                raise ValueError(f"{name}: unpacked weights differ from the model's")
            bias = torch.from_numpy(unpack_half(blob(entry["bias"]), (rows,)).copy())
            if not torch.equal(bias, modules[name].bias.data):
                raise ValueError(f"{name}: unpacked bias differs from the model's")
        elif name in parameters:
            values = torch.from_numpy(unpack_half(blob(entry["data"]), entry["shape"]).copy())
            if not torch.equal(values, parameters[name].data.reshape(values.shape)):
                raise ValueError(f"{name}: unpacked values differ from the model's")
    for index, adapter in enumerate(head.adapters):
        for part in ("qkv", "proj", "fc1", "fc2"):
            for side in ("down", "up"):
                entry = manifest["tensors"][f"slots.adapters.{index}.{part}.{side}"]
                values = torch.from_numpy(unpack_half(blob(entry["data"]), entry["shape"]).copy())
                if not torch.equal(values, getattr(adapter[part], side).weight.data):
                    raise ValueError(f"adapter {index}.{part}.{side} differs")


class SingleCrop(SlotDetector):
    """The slot detector as the browser runs it: float32, one 378 x 378 crop."""

    def crop(self, image):
        module = type(self.inner).__module__.rsplit(".", 1)[0]
        crops_module = importlib.import_module(f"{module}.image_crops")
        config = self.inner.config.vision
        made = crops_module.overlap_crop_image(
            np.array(image.convert("RGB")), max_crops=1, overlap_margin=config.overlap_margin
        )
        if made["tiling"] != (1, 1) or not (made["crops"][0] == made["crops"][1]).all():
            raise ValueError("With one crop the local crop should equal the global crop")
        return made["crops"][0]

    def features(self, crop):
        """Vision encoder output [729, 1152] and the projected embeddings [729, 2048]."""
        pixels = torch.from_numpy(crop).permute(2, 0, 1)[None].float().div(255).sub(0.5).div(0.5)
        out = self.inner._vis_enc(pixels)[0]
        side = self.inner.config.vision.enc_n_layers
        return out, self.inner._vis_proj(out, out.view(side, side, -1))

    def _encode(self, images):
        return torch.stack([self.features(self.crop(image))[1] for image in images])


def raw(path, array, dtype):
    path.write_bytes(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return {"file": path.name, "dtype": np.dtype(dtype).name, "shape": list(array.shape)}


@torch.inference_mode()
def reference(detector, image, folder):
    """Write one image's crop, activations, slots and selected boxes; returns their index."""
    folder.mkdir(parents=True, exist_ok=True)
    inner, head = detector.inner, detector.head
    crop = detector.crop(image)
    vision, projection = detector.features(crop)
    trace = {}
    cache = image_cache(inner, detector.functions, projection[None], trace=trace)
    objectness, box_logits, logits = head(
        inner.text, inner.region, cache, detector.functions.apply_rotary_emb, trace=trace
    )
    bin_probability, bins = box_logits.softmax(-1).max(-1)
    token_probability, tokens = logits.softmax(-1).max(-1)
    arrays = {
        "crop": raw(folder / "crop.u8", crop, np.uint8),
        "vision": raw(folder / "vision.f32", vision.numpy(), np.float32),
        "projection": raw(folder / "projection.f32", projection.numpy(), np.float32),
        "slots": raw(folder / "slots.f32", trace["slots"][0].flatten(0, 1).numpy(), np.float32),
        "objectness": raw(folder / "objectness.f32", objectness[0].numpy(), np.float32),
        "bins": raw(folder / "bins.i32", bins[0].numpy(), np.int32),
        "bin_probability": raw(folder / "bin_probability.f32", bin_probability[0], np.float32),
        "tokens": raw(folder / "tokens.i32", tokens[0].numpy(), np.int32),
        "token_probability": raw(
            folder / "token_probability.f32", token_probability[0], np.float32
        ),
    }
    for block in TRACED_PREFIX_BLOCKS:
        hidden = trace["prefix"][block][0].numpy()
        arrays[f"prefix_{block:02d}"] = raw(folder / f"prefix_{block:02d}.f32", hidden, np.float32)
    result = detector.detect(image, COCO_NAMES)
    found = [
        {"label": name, "box": box, "score": score}
        for name, boxes, scores in zip(COCO_NAMES, result["boxes"], result["scores"], strict=True)
        for box, score in zip(boxes, scores, strict=True)
    ]
    index = {
        "width": image.width,
        "height": image.height,
        "arrays": arrays,
        "slots": result["slots"],
        "boxes": found,
    }
    (folder / "reference.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True, help="head.pt of a slot training run")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--component-bits", action="append", default=[], metavar="NAME=BITS")
    parser.add_argument("--threshold", type=float, required=True, help="selection score threshold")
    parser.add_argument("--images", nargs="*", default=[], type=Path)
    args = parser.parse_args()

    bits_of = dict.fromkeys(COMPONENTS, args.bits)
    for item in args.component_bits:
        name, bits = item.split("=")
        if name not in bits_of:
            raise ValueError(f"Unknown component {name!r}")
        bits_of[name] = int(bits)

    state = torch.load(args.head, map_location="cpu")
    backbone = load_backbone(**state["config"]["model"], device="cpu")
    inner = backbone[1].float()
    shards = Shards(args.out)
    tensors = export_backbone(inner, shards, bits_of)
    detector = SingleCrop(
        args.head, "cpu", threshold=args.threshold, all_slots=True, tiles=1, backbone=backbone
    )
    detector.head.float()
    tensors |= export_head(detector.head, inner, shards)
    shards.close()
    config = state["config"]
    manifest = {
        "model": config["model"],
        "head": {
            "run": Path(args.head).parent.name,
            "slots": config["slots"],
            "label_tokens": config["label_tokens"],
            "rank": config["rank"],
            "trained_weight_bits": config.get("weight_bits"),
            "trained_max_crops": config.get("max_crops"),
        },
        "bits": bits_of,
        "group": GROUP,
        "selection": {
            "threshold": args.threshold,
            "duplicate_iou": detector.duplicate_iou,
            "box_power": detector.box_power,
            "coordinates": COORDINATES,
        },
        "vocabulary": detector.vocabulary,
        "shards": shards.files,
        "tensors": tensors,
        "reference": [image.stem for image in args.images],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    verify(args.out, manifest, inner, detector.head)
    for path in args.images:
        reference(detector, Image.open(path), args.out / "reference" / path.stem)


if __name__ == "__main__":
    main()
