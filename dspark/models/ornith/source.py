"""The uncompressed "source" checkpoint format Ornith is exported in
before conversion to dspark's quantized, memory-mapped .dsq format.

This stands in for whatever full-precision export format an Ornith
training run would produce (analogous to a HF ``pytorch_model.bin``): a
JSON header describing tensor shapes/offsets followed by a raw float32
blob. It is intentionally simple and stdlib-only.
"""

import json
import mmap
import struct
from array import array

MAGIC = b"ORNSRC01"


def write_source_checkpoint(path, config, tensors):
    """Write a source checkpoint.

    ``tensors`` maps tensor name -> ``(shape, flat_values)``.
    """
    offset = 0
    meta = {}
    blob_parts = []
    for name, (shape, values) in tensors.items():
        raw = array("f", values).tobytes()
        meta[name] = {"shape": list(shape), "offset": offset, "nbytes": len(raw)}
        blob_parts.append(raw)
        offset += len(raw)
    header = json.dumps({"config": config, "tensors": meta}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">I", len(header)))
        f.write(header)
        for part in blob_parts:
            f.write(part)


class OrnithSourceCheckpoint:
    """Memory-mapped reader for the uncompressed source format.

    Conversion reads one tensor at a time through this class so even the
    "before quantization" step never needs the full fp32 model resident at
    once.
    """

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        magic = self._file.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a valid Ornith source checkpoint")
        header_len = struct.unpack(">I", self._file.read(4))[0]
        header = json.loads(self._file.read(header_len))
        self.config = header["config"]
        self._tensor_meta = header["tensors"]
        self._blob_start = len(MAGIC) + 4 + header_len
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor_names(self):
        return list(self._tensor_meta.keys())

    def read_tensor(self, name):
        meta = self._tensor_meta[name]
        start = self._blob_start + meta["offset"]
        raw = self._mm[start : start + meta["nbytes"]]
        values = array("f")
        values.frombytes(raw)
        return meta["shape"], list(values)

    def close(self):
        self._mm.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
