"""Convert an uncompressed Ornith source checkpoint into dspark's
quantized, memory-mapped .dsq format.

Tables addressed per-row at inference time (the token embedding and, if
untied, the lm_head) are quantized with exactly one block per row so that
:meth:`dspark.storage.LazyTensor.get_row` can decode a single row in
isolation. Every other weight matrix uses a configurable block size, since
those tensors are decoded a whole layer at a time anyway.
"""

from dspark.quant import quantize_blockwise_int4, quantize_blockwise_int8
from dspark.storage import DsparkWriter

from .source import OrnithSourceCheckpoint

#: Tensors that must be quantized with one block per row so a single row
#: can be decoded without touching the rest of the tensor.
ROW_ADDRESSED_TENSORS = {"embed_tokens.weight", "lm_head.weight"}

_QUANTIZERS = {
    "int8": quantize_blockwise_int8,
    "int4": quantize_blockwise_int4,
}


def convert_source_to_dspark(src_path, out_path, quant="int8", block_size=64):
    if quant not in _QUANTIZERS:
        raise ValueError(f"unsupported quant {quant!r}, expected one of {sorted(_QUANTIZERS)}")
    quantize = _QUANTIZERS[quant]

    with OrnithSourceCheckpoint(src_path) as src:
        writer = DsparkWriter(model_family="ornith", config=src.config)
        for name in src.tensor_names():
            shape, values = src.read_tensor(name)
            bs = shape[-1] if name in ROW_ADDRESSED_TENSORS else block_size
            data, scales = quantize(values, bs)
            writer.add_tensor(name, quant, shape, bs, data, scales)
        writer.save(out_path)
    return out_path
