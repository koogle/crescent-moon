"""Simulated weight quantisation: how much accuracy low-bit weights would cost.

`round_weights_` replaces every `nn.Linear` weight under a module, in place, by its
symmetric round-to-nearest quantisation to `bits`-bit integers with one scale per group
of `group` consecutive input weights (a row's last group may be shorter), dequantised back to the weight's dtype. This is
the arithmetic of block-wise integer weight formats (such as 4-bit blocks of 32), so
evaluating with it measures their accuracy; it does not make inference smaller or faster.
"""

import torch


def quantise(weight, bits, group=32, scale_dtype=torch.float32):
    """Integers [rows, columns] (int8) and scales [rows, groups] of a weight matrix:
    symmetric round to nearest, one scale per group of `group` consecutive input
    weights, the scale held in `scale_dtype` (FP16 for packed formats)."""
    if bits < 2:
        raise ValueError(f"bits must be at least 2, not {bits}")
    top = 2 ** (bits - 1) - 1
    rows, columns = weight.shape
    padded = -columns % group  # zeros change no group's largest magnitude
    blocks = torch.nn.functional.pad(weight.float(), (0, padded)).view(rows, -1, group)
    scale = (blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / top).to(scale_dtype).float()
    integers = (blocks / scale).round().clamp(-top - 1, top)
    return integers.view(rows, -1)[:, :columns].to(torch.int8), scale.squeeze(-1)


def dequantise(integers, scale, group=32):
    """The float32 weights that `quantise`'s integers and scales stand for."""
    return integers.float() * scale.repeat_interleave(group, dim=1)[:, : integers.size(1)]


def round_weights_(module, bits, group=32):
    """Quantise and dequantise every Linear weight under `module`; returns the number of
    layers and weights changed."""
    layers = weights = 0
    for layer in module.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        weight = layer.weight.data
        layer.weight.data = dequantise(*quantise(weight, bits, group), group).to(weight.dtype)
        layers += 1
        weights += weight.numel()
    if not layers:
        raise ValueError("No Linear layers to quantise")
    return layers, weights
