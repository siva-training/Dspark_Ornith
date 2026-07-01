"""The .dsq (dspark-quantized) checkpoint container format.

Layout on disk::

    b"DSPK"                  4 bytes magic
    uint32 big-endian        header length in bytes
    header (utf-8 json)      {"format_version", "model_family", "config",
                               "tensors": {name: {dtype, shape, block_size,
                                                   data_offset, data_nbytes,
                                                   scales_offset, scales_nbytes}}}
    <tensor blob region>     concatenated quantized bytes + scale arrays

The whole file is opened with :mod:`mmap`, so the OS page cache - not the
Python process - holds the bulk of the model. Only the small JSON header is
read eagerly; individual tensors (or even individual rows, for embedding /
lm_head tables) are decoded on demand via :class:`LazyTensor`, which is how
a checkpoint far larger than physical RAM can still be "loaded".
"""

import json
import math
import mmap
import os
import struct
from array import array

from .quant import dequantize_blockwise_int4, dequantize_blockwise_int8

MAGIC = b"DSPK"
FORMAT_VERSION = 1


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class DsparkWriter:
    """Accumulates quantized tensors in memory and writes a .dsq file.

    Tensors are added one at a time (already quantized), so the writer
    itself never needs to hold more than one full-precision tensor's worth
    of memory during conversion.
    """

    def __init__(self, model_family, config):
        self.model_family = model_family
        self.config = dict(config)
        self._tensors = {}

    def add_tensor(self, name, dtype, shape, block_size, data, scales):
        if dtype not in ("int8", "int4"):
            raise ValueError(f"unsupported dtype {dtype!r}")
        self._tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "block_size": block_size,
            "data": bytes(data),
            "scales": array("f", scales).tobytes(),
        }

    def save(self, path):
        offset = 0
        meta = {}
        blob_parts = []
        for name, t in self._tensors.items():
            data_offset = offset
            blob_parts.append(t["data"])
            offset += len(t["data"])
            scales_offset = offset
            blob_parts.append(t["scales"])
            offset += len(t["scales"])
            meta[name] = {
                "dtype": t["dtype"],
                "shape": t["shape"],
                "block_size": t["block_size"],
                "data_offset": data_offset,
                "data_nbytes": len(t["data"]),
                "scales_offset": scales_offset,
                "scales_nbytes": len(t["scales"]),
            }
        header = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "model_family": self.model_family,
                "config": self.config,
                "tensors": meta,
            }
        ).encode("utf-8")
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack(">I", len(header)))
            f.write(header)
            for part in blob_parts:
                f.write(part)


class LazyTensor:
    """A handle to one tensor inside a memory-mapped .dsq file.

    Nothing is decoded until :meth:`load` (whole tensor) or :meth:`get_row`
    (single row - used for huge embedding/lm_head tables) is called.
    """

    def __init__(self, mm, blob_start, meta):
        self._mm = mm
        self._blob_start = blob_start
        self._meta = meta
        self._cache = None

    @property
    def dtype(self):
        return self._meta["dtype"]

    @property
    def shape(self):
        return tuple(self._meta["shape"])

    @property
    def block_size(self):
        return self._meta["block_size"]

    def numel(self):
        return _numel(self._meta["shape"])

    def on_disk_nbytes(self):
        return self._meta["data_nbytes"] + self._meta["scales_nbytes"]

    def load(self):
        """Decode the entire tensor to a flat list[float] (row-major)."""
        if self._cache is not None:
            return self._cache
        meta = self._meta
        data_start = self._blob_start + meta["data_offset"]
        data = self._mm[data_start : data_start + meta["data_nbytes"]]
        scales_start = self._blob_start + meta["scales_offset"]
        scales_bytes = self._mm[scales_start : scales_start + meta["scales_nbytes"]]
        scales = array("f")
        scales.frombytes(scales_bytes)
        count = self.numel()
        if meta["dtype"] == "int8":
            values = dequantize_blockwise_int8(data, scales, meta["block_size"], count)
        else:
            values = dequantize_blockwise_int4(data, scales, count, meta["block_size"])
        self._cache = values
        return values

    def get_row(self, row_idx):
        """Decode a single row of a 2D tensor without touching the rest.

        Requires the tensor to have been quantized with one block per row
        (``block_size == shape[1]``), which is how dspark stores embedding
        and lm_head tables - this keeps a single token lookup's cost
        independent of vocab size.
        """
        meta = self._meta
        if len(meta["shape"]) != 2:
            raise ValueError("get_row requires a 2D tensor")
        in_features = meta["shape"][1]
        if meta["block_size"] != in_features:
            raise ValueError(
                "get_row requires one quantization block per row "
                f"(block_size={meta['block_size']} != in_features={in_features})"
            )
        if meta["dtype"] == "int8":
            row_nbytes = in_features
        else:
            row_nbytes = (in_features + 1) // 2
        data_start = self._blob_start + meta["data_offset"] + row_idx * row_nbytes
        row_bytes = self._mm[data_start : data_start + row_nbytes]
        scale_start = self._blob_start + meta["scales_offset"] + row_idx * 4
        scale = struct.unpack("<f", self._mm[scale_start : scale_start + 4])[0]
        if meta["dtype"] == "int8":
            return dequantize_blockwise_int8(row_bytes, [scale], in_features, in_features)
        return dequantize_blockwise_int4(row_bytes, [scale], in_features, in_features)

    def release(self):
        """Drop any decoded cache, allowing the memory to be reclaimed."""
        self._cache = None


class DsparkCheckpoint:
    """Read-only, memory-mapped view over a .dsq file."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        try:
            magic = self._file.read(4)
            if magic != MAGIC:
                raise ValueError(f"{path} is not a valid dspark (.dsq) checkpoint")
            header_len = struct.unpack(">I", self._file.read(4))[0]
            header = json.loads(self._file.read(header_len))
            self.format_version = header["format_version"]
            self.model_family = header["model_family"]
            self.config = header["config"]
            self._tensor_meta = header["tensors"]
            self._blob_start = 4 + 4 + header_len
            size = os.fstat(self._file.fileno()).st_size
            if size <= self._blob_start:
                # mmap() cannot map a zero-length region; only possible for a
                # checkpoint with no tensors at all.
                self._mm = None
            else:
                self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

    def tensor_names(self):
        return list(self._tensor_meta.keys())

    def get_tensor(self, name):
        if name not in self._tensor_meta:
            raise KeyError(f"no tensor named {name!r} in {self.path}")
        return LazyTensor(self._mm, self._blob_start, self._tensor_meta[name])

    def file_size_bytes(self):
        return os.fstat(self._file.fileno()).st_size

    def close(self):
        if self._mm is not None:
            self._mm.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
