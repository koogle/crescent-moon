"""The packed weight format the browser runtime reads (`web/`).

A quantised matrix [rows, columns] is stored as two blobs:

- integers: row-major, each row padded with zeros to a multiple of 32 columns, as
  two's-complement fields of `bits` bits, 32 // bits fields per little-endian u32 word,
  the first field in the lowest bits;
- scales: one FP16 per group of 32 columns, row-major, each row padded to an even number
  of scales so that a row is whole u32 words (the first scale in the low half).

A dense matrix is row-major FP16 with rows padded to an even length. Every blob starts
on a 4-byte boundary of its shard; `Shards` writes them and records where they went.
"""

import hashlib
from pathlib import Path

import numpy as np

GROUP = 32


def pack_integers(integers, bits):
    """Bytes of an int8 matrix [rows, columns] in `bits`-bit fields; see the module text."""
    rows, columns = integers.shape
    per_word = 32 // bits
    padded = np.zeros((rows, columns + -columns % GROUP), dtype=np.uint32)
    padded[:, :columns] = integers.astype(np.int32) & ((1 << bits) - 1)
    fields = padded.reshape(rows, -1, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)[None, None, :]
    return (fields << shifts).sum(-1, dtype=np.uint32).astype("<u4").tobytes()


def unpack_integers(data, rows, columns, bits):
    """The int8 matrix `pack_integers` stored."""
    per_word = 32 // bits
    words = np.frombuffer(data, dtype="<u4").reshape(rows, -1, 1)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)[None, None, :]
    fields = ((words >> shifts) & ((1 << bits) - 1)).astype(np.int32)
    fields = np.where(fields >= 1 << (bits - 1), fields - (1 << bits), fields)
    return fields.reshape(rows, -1)[:, :columns].astype(np.int8)


def pack_half(values):
    """Bytes of a float array as FP16, its last axis padded with zeros to an even length."""
    values = np.atleast_1d(np.asarray(values, dtype=np.float16))
    if values.shape[-1] % 2:
        values = np.concatenate([values, np.zeros_like(values[..., :1])], axis=-1)
    return values.astype("<f2").tobytes()


def unpack_half(data, shape):
    """The float32 array `pack_half` stored for an array of `shape`."""
    stride = shape[-1] + shape[-1] % 2
    values = np.frombuffer(data, dtype="<f2").reshape(*shape[:-1], stride)
    return values[..., : shape[-1]].astype(np.float32)


class Shards:
    """Writes blobs into files of at most `limit` bytes under a folder."""

    def __init__(self, folder, limit=256 * 2**20):
        self.folder, self.limit = Path(folder), limit
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files, self.handle, self.used = [], None, 0

    def add(self, data):
        """Append one blob; returns its {"shard", "offset", "bytes", "sha256"}."""
        if len(data) % 4:
            raise ValueError("Blobs are whole u32 words")
        if self.handle is None or self.used + len(data) > self.limit:
            self.close()
            self.files.append(f"weights-{len(self.files):03d}.bin")
            self.handle, self.used = (self.folder / self.files[-1]).open("wb"), 0
        place = {
            "shard": len(self.files) - 1,
            "offset": self.used,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        self.handle.write(data)
        self.used += len(data)
        return place

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
