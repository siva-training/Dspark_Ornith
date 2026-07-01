"""Blockwise weight quantization primitives.

dspark keeps model weights on disk (and mmap'd) as int8 or int4 blockwise
quantized values with one float32 scale per block. This is the same family
of technique used by GGUF/llama.cpp-style consumer-hardware runtimes: it
lets a checkpoint many times larger than available RAM be memory-mapped and
decoded only a block/row/layer at a time, instead of ever materializing the
whole model as float32 in memory.

The functions here operate on flat Python sequences of floats and are
stdlib-only (no numpy/torch) so dspark itself has effectively no install
footprint - fitting for "run on consumer hardware".
"""

import math

INT8_MAX = 127
INT4_MAX = 7


def _clip(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _block_ranges(n, block_size):
    start = 0
    while start < n:
        end = min(start + block_size, n)
        yield start, end
        start = end


def quantize_blockwise_int8(values, block_size):
    """Quantize ``values`` to signed int8 with one scale per block.

    Returns ``(data: bytes, scales: list[float])``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n = len(values)
    data = bytearray(n)
    scales = []
    for start, end in _block_ranges(n, block_size):
        amax = 0.0
        for i in range(start, end):
            v = values[i]
            if v < 0:
                v = -v
            if v > amax:
                amax = v
        scale = (amax / INT8_MAX) if amax > 0.0 else 1.0
        inv_scale = 1.0 / scale
        for i in range(start, end):
            q = int(round(values[i] * inv_scale))
            q = _clip(q, -INT8_MAX, INT8_MAX)
            data[i] = q & 0xFF
        scales.append(scale)
    return bytes(data), scales


def dequantize_blockwise_int8(data, scales, block_size, count=None):
    """Inverse of :func:`quantize_blockwise_int8`."""
    n = count if count is not None else len(data)
    out = [0.0] * n
    for block_idx, (start, end) in enumerate(_block_ranges(n, block_size)):
        scale = scales[block_idx]
        for i in range(start, end):
            b = data[i]
            signed = b - 256 if b > 127 else b
            out[i] = signed * scale
    return out


def quantize_blockwise_int4(values, block_size):
    """Quantize ``values`` to signed 4-bit values, two packed per byte.

    Returns ``(data: bytes, scales: list[float])``. ``data`` has
    ``ceil(len(values) / 2)`` bytes.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n = len(values)
    packed = bytearray((n + 1) // 2)
    scales = []
    for start, end in _block_ranges(n, block_size):
        amax = 0.0
        for i in range(start, end):
            v = values[i]
            if v < 0:
                v = -v
            if v > amax:
                amax = v
        scale = (amax / INT4_MAX) if amax > 0.0 else 1.0
        inv_scale = 1.0 / scale
        for i in range(start, end):
            q = int(round(values[i] * inv_scale))
            q = _clip(q, -INT4_MAX, INT4_MAX)
            nibble = q & 0xF
            byte_idx = i // 2
            if i % 2 == 0:
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | nibble
            else:
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (nibble << 4)
        scales.append(scale)
    return bytes(packed), scales


def dequantize_blockwise_int4(data, scales, count, block_size):
    """Inverse of :func:`quantize_blockwise_int4`. ``count`` is required
    since a packed byte cannot tell us if a trailing nibble is padding."""
    out = [0.0] * count
    for block_idx, (start, end) in enumerate(_block_ranges(count, block_size)):
        scale = scales[block_idx]
        for i in range(start, end):
            byte = data[i // 2]
            nibble = byte & 0xF if i % 2 == 0 else (byte >> 4) & 0xF
            signed = nibble - 16 if nibble > 7 else nibble
            out[i] = signed * scale
    return out


def bytes_per_value(dtype):
    """Approximate on-disk bytes/value for a quantization dtype, excluding
    the amortized per-block scale overhead. Used for footprint reporting."""
    if dtype == "int8":
        return 1.0
    if dtype == "int4":
        return 0.5
    if dtype == "float32":
        return 4.0
    raise ValueError(f"unknown dtype {dtype!r}")


def quantized_size_bytes(numel, dtype, block_size):
    """Total on-disk size (data + scales) for ``numel`` values."""
    num_blocks = math.ceil(numel / block_size)
    data_bytes = math.ceil(numel * bytes_per_value(dtype))
    scale_bytes = num_blocks * 4  # float32 scale per block
    return data_bytes + scale_bytes
