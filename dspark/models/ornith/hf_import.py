"""Import Ornith weights straight from a Hugging Face-style checkpoint
directory (a ``config.json`` plus one or more ``.safetensors`` files) into
dspark's quantized, memory-mapped ``.dsq`` format - without depending on
``torch``, ``transformers``, or even the ``safetensors`` pip package.

Only the safetensors container is supported (not ``pytorch_model.bin``):

- Its layout - an 8-byte header length, a small JSON header, then a raw
  tensor byte blob - is simple enough to parse with :mod:`struct` and
  :mod:`json` alone, and can be memory-mapped so conversion reads one
  tensor at a time instead of ever holding the whole checkpoint resident.
- Unlike ``pytorch_model.bin`` (a pickle archive), loading it never
  executes arbitrary code, which matters when pulling checkpoints from
  the internet.

Hugging Face checkpoints in this architecture family conventionally name
tensors the same way LLaMA/Mistral do (``model.embed_tokens.weight``,
``model.layers.<i>.self_attn.q_proj.weight``, ...); :data:`DEFAULT_HF_NAME_MAP`
assumes that convention. If a specific Ornith release names things
differently, pass a custom ``name_map`` to :func:`convert_hf_to_dspark`
(or edit the default) - :func:`convert_hf_to_dspark` also fails fast,
listing exactly which expected tensors and config fields it could not
find, so a naming mismatch is easy to diagnose and fix.
"""

import json
import mmap
import os
import struct

from dspark.quant import quantize_blockwise_int4, quantize_blockwise_int8
from dspark.storage import DsparkWriter

from .config import OrnithConfig

#: Hugging Face LLaMA-style config.json field -> OrnithConfig field.
#: Extend/override this if a specific Ornith config.json uses different keys.
HF_CONFIG_FIELD_MAP = {
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_layers",
    "num_attention_heads": "num_heads",
    "num_key_value_heads": "num_kv_heads",
    "vocab_size": "vocab_size",
    "intermediate_size": "intermediate_size",
    "max_position_embeddings": "max_position_embeddings",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "tie_word_embeddings": "tie_word_embeddings",
}

_REQUIRED_ORNITH_FIELDS = {"hidden_size", "num_layers", "num_heads", "vocab_size", "intermediate_size"}

#: dspark tensor name (``{i}`` is a layer index placeholder) -> Hugging
#: Face tensor name. Assumes the common LLaMA-style naming convention;
#: pass ``name_map=`` to :func:`convert_hf_to_dspark` to override.
DEFAULT_HF_NAME_MAP = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "layers.{i}.input_layernorm.weight": "model.layers.{i}.input_layernorm.weight",
    "layers.{i}.self_attn.q_proj.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "layers.{i}.self_attn.k_proj.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "layers.{i}.self_attn.v_proj.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "layers.{i}.self_attn.o_proj.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "layers.{i}.post_attention_layernorm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "layers.{i}.mlp.gate_proj.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "layers.{i}.mlp.up_proj.weight": "model.layers.{i}.mlp.up_proj.weight",
    "layers.{i}.mlp.down_proj.weight": "model.layers.{i}.mlp.down_proj.weight",
}

#: Tensors quantized one block per row so a single row can be decoded
#: without touching the rest of the tensor (see dspark.storage.LazyTensor).
ROW_ADDRESSED_TENSORS = {"embed_tokens.weight", "lm_head.weight"}

_QUANTIZERS = {"int8": quantize_blockwise_int8, "int4": quantize_blockwise_int4}


def _bf16_bytes_to_float(raw2):
    # bfloat16 is exactly the top 16 bits of an IEEE754 float32; padding
    # the low 16 bits with zero and reinterpreting as float32 reconstructs
    # the value (with the precision bf16 already discarded).
    return struct.unpack("<f", b"\x00\x00" + raw2)[0]


def _unpack_f32(buf):
    return list(struct.unpack(f"<{len(buf) // 4}f", buf))


def _unpack_f16(buf):
    return list(struct.unpack(f"<{len(buf) // 2}e", buf))


def _unpack_bf16(buf):
    return [_bf16_bytes_to_float(buf[i : i + 2]) for i in range(0, len(buf), 2)]


_DTYPE_UNPACKERS = {"F32": _unpack_f32, "F16": _unpack_f16, "BF16": _unpack_bf16}


def ornith_config_from_hf(hf_config):
    """Build an :class:`OrnithConfig` from a Hugging Face ``config.json`` dict."""
    kwargs = {
        dspark_key: hf_config[hf_key]
        for hf_key, dspark_key in HF_CONFIG_FIELD_MAP.items()
        if hf_key in hf_config
    }
    if "num_kv_heads" not in kwargs and "num_heads" in kwargs:
        kwargs["num_kv_heads"] = kwargs["num_heads"]  # plain multi-head attention, no GQA
    missing = _REQUIRED_ORNITH_FIELDS - kwargs.keys()
    if missing:
        raise ValueError(
            f"config.json is missing fields dspark needs for Ornith: {sorted(missing)}. "
            "If this checkpoint uses different key names, extend HF_CONFIG_FIELD_MAP."
        )
    return OrnithConfig(**kwargs)


class SafetensorsFile:
    """Memory-mapped reader for a single ``.safetensors`` file."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        try:
            header_len = struct.unpack("<Q", self._file.read(8))[0]
            header = json.loads(self._file.read(header_len))
            header.pop("__metadata__", None)
            self._tensor_meta = header
            self._data_start = 8 + header_len
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

    def tensor_names(self):
        return list(self._tensor_meta.keys())

    def read_tensor(self, name):
        meta = self._tensor_meta[name]
        dtype = meta["dtype"]
        if dtype not in _DTYPE_UNPACKERS:
            raise ValueError(
                f"unsupported safetensors dtype {dtype!r} for tensor {name!r} "
                f"(supported: {sorted(_DTYPE_UNPACKERS)})"
            )
        start, end = meta["data_offsets"]
        buf = self._mm[self._data_start + start : self._data_start + end]
        return meta["shape"], _DTYPE_UNPACKERS[dtype](buf)

    def close(self):
        self._mm.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class HFCheckpointDir:
    """A Hugging Face checkpoint directory: ``config.json`` plus either a
    single ``model.safetensors`` or a sharded set described by
    ``model.safetensors.index.json``."""

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        config_path = os.path.join(hf_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"no config.json found in {hf_dir}")
        with open(config_path) as f:
            self.hf_config = json.load(f)
        self._open_files = {}
        self._name_to_file = {}

        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self._name_to_file = dict(index["weight_map"])
        else:
            single_path = os.path.join(hf_dir, "model.safetensors")
            if not os.path.exists(single_path):
                raise FileNotFoundError(
                    f"no model.safetensors or model.safetensors.index.json found in {hf_dir}"
                )
            for name in self._get_file("model.safetensors").tensor_names():
                self._name_to_file[name] = "model.safetensors"

    def _get_file(self, filename):
        if filename not in self._open_files:
            self._open_files[filename] = SafetensorsFile(os.path.join(self.hf_dir, filename))
        return self._open_files[filename]

    def tensor_names(self):
        return list(self._name_to_file.keys())

    def read_tensor(self, hf_name):
        if hf_name not in self._name_to_file:
            raise KeyError(f"tensor {hf_name!r} not found under {self.hf_dir}")
        return self._get_file(self._name_to_file[hf_name]).read_tensor(hf_name)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def convert_hf_to_dspark(hf_dir, out_path, quant="int8", block_size=64, name_map=None):
    """Quantize a Hugging Face Ornith checkpoint directory directly into a
    dspark ``.dsq`` file, one tensor at a time (no full-precision copy of
    the model is ever held in memory at once)."""
    if quant not in _QUANTIZERS:
        raise ValueError(f"unsupported quant {quant!r}, expected one of {sorted(_QUANTIZERS)}")
    quantize = _QUANTIZERS[quant]
    name_map = name_map or DEFAULT_HF_NAME_MAP

    with HFCheckpointDir(hf_dir) as hf:
        config = ornith_config_from_hf(hf.hf_config)
        writer = DsparkWriter(model_family="ornith", config=config.to_dict())

        def convert_one(dspark_name, hf_name):
            try:
                shape, values = hf.read_tensor(hf_name)
            except KeyError:
                raise KeyError(
                    f"expected tensor {hf_name!r} (for dspark tensor {dspark_name!r}) not found in "
                    f"{hf_dir}. Available tensors include: {sorted(hf.tensor_names())[:10]}... "
                    "If this checkpoint uses different names, pass a custom name_map."
                ) from None
            bs = shape[-1] if dspark_name in ROW_ADDRESSED_TENSORS else block_size
            data, scales = quantize(values, bs)
            writer.add_tensor(dspark_name, quant, shape, bs, data, scales)

        convert_one("embed_tokens.weight", name_map["embed_tokens.weight"])
        for layer in range(config.num_layers):
            for dspark_template, hf_template in name_map.items():
                if "{i}" not in dspark_template:
                    continue
                convert_one(dspark_template.format(i=layer), hf_template.format(i=layer))
        convert_one("norm.weight", name_map["norm.weight"])
        if not config.tie_word_embeddings:
            convert_one("lm_head.weight", name_map["lm_head.weight"])

        writer.save(out_path)
    return out_path
