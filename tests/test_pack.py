import numpy as np
import pytest
import torch

from crescent_moon.pack import Shards, pack_half, pack_integers, unpack_half, unpack_integers
from crescent_moon.quantize import dequantise, quantise, round_weights_


@pytest.mark.parametrize(("bits", "columns"), [(4, 70), (8, 70), (4, 64)])
def test_packed_integers_unpack_to_the_same_values(bits, columns):
    rng = np.random.default_rng(bits * columns)
    top = 2 ** (bits - 1)
    integers = rng.integers(-top, top, size=(5, columns), dtype=np.int8)
    data = pack_integers(integers, bits)
    assert len(data) == 5 * (columns + -columns % 32) * bits // 8
    assert (unpack_integers(data, 5, columns, bits) == integers).all()


def test_first_field_is_in_the_lowest_bits():
    integers = np.zeros((1, 32), dtype=np.int8)
    integers[0, 0], integers[0, 1] = -1, 3
    words = np.frombuffer(pack_integers(integers, 4), dtype="<u4")
    assert words[0] == 0x3F


def test_half_blobs_round_trip_with_an_odd_row_length():
    values = np.linspace(-2, 2, 15, dtype=np.float32).reshape(3, 5)
    data = pack_half(values)
    assert len(data) == 3 * 6 * 2
    assert np.allclose(unpack_half(data, (3, 5)), values.astype(np.float16))


def test_quantise_with_half_scales_matches_the_rounding_up_to_scale_precision():
    torch.manual_seed(2)
    layer = torch.nn.Linear(70, 6)
    weight = layer.weight.detach().clone()
    integers, scale = quantise(weight, 4, scale_dtype=torch.float16)
    assert (scale == scale.half().float()).all()
    round_weights_(layer, 4)
    assert torch.allclose(dequantise(integers, scale), layer.weight, atol=2e-3)


def test_shards_roll_over_at_the_limit_and_record_places(tmp_path):
    shards = Shards(tmp_path, limit=16)
    first, second, third = shards.add(b"a" * 8), shards.add(b"b" * 8), shards.add(b"c" * 12)
    shards.close()
    assert (first["shard"], second["offset"], third["shard"], third["offset"]) == (0, 8, 1, 0)
    assert (tmp_path / "weights-000.bin").read_bytes() == b"a" * 8 + b"b" * 8
    assert (tmp_path / "weights-001.bin").read_bytes() == b"c" * 12
