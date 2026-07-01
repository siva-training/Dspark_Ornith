import json
import os
import struct
import tempfile
import unittest
from array import array

from dspark.models.ornith.config import OrnithConfig
from dspark.models.ornith.convert import convert_source_to_dspark
from dspark.models.ornith.hf_import import (
    DEFAULT_HF_NAME_MAP,
    convert_hf_to_dspark,
    ornith_config_from_hf,
)
from dspark.models.ornith.model import OrnithForCausalLM
from dspark.models.ornith.source import write_source_checkpoint
from dspark.models.ornith.synthetic import generate_synthetic_ornith_tensors
from dspark.storage import DsparkCheckpoint


def write_safetensors(path, tensors, dtype="F32"):
    """Minimal safetensors writer, F32 only, for building test fixtures."""
    assert dtype == "F32"
    header = {}
    blobs = []
    offset = 0
    for name, (shape, values) in tensors.items():
        raw = array("f", values).tobytes()
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def dspark_template_to_hf_name(template, layer=None):
    hf_template = DEFAULT_HF_NAME_MAP[template]
    return hf_template.format(i=layer) if layer is not None else hf_template


class TestHFImport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.config = OrnithConfig(
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=32,
            intermediate_size=24,
            max_position_embeddings=64,
        )
        self.tensors = generate_synthetic_ornith_tensors(self.config, seed=99)

        self.hf_dir = os.path.join(self.tmpdir.name, "hf_ornith")
        os.makedirs(self.hf_dir)
        hf_config = {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_layers,
            "num_attention_heads": self.config.num_heads,
            "num_key_value_heads": self.config.num_kv_heads,
            "vocab_size": self.config.vocab_size,
            "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rope_theta": self.config.rope_theta,
            "rms_norm_eps": self.config.rms_norm_eps,
            "tie_word_embeddings": self.config.tie_word_embeddings,
            "model_type": "ornith",
        }
        with open(os.path.join(self.hf_dir, "config.json"), "w") as f:
            json.dump(hf_config, f)

        hf_tensors = {}
        for name, value in self.tensors.items():
            if name.startswith("layers."):
                layer = int(name.split(".")[1])
                rest = name.split(".", 2)[2]
                hf_name = dspark_template_to_hf_name(f"layers.{{i}}.{rest}", layer=layer)
            else:
                hf_name = dspark_template_to_hf_name(name)
            hf_tensors[hf_name] = value
        write_safetensors(os.path.join(self.hf_dir, "model.safetensors"), hf_tensors)

    def test_ornith_config_from_hf(self):
        with open(os.path.join(self.hf_dir, "config.json")) as f:
            hf_config = json.load(f)
        config = ornith_config_from_hf(hf_config)
        self.assertEqual(config.hidden_size, self.config.hidden_size)
        self.assertEqual(config.num_layers, self.config.num_layers)
        self.assertEqual(config.num_kv_heads, self.config.num_kv_heads)

    def test_missing_config_fields_raise_clear_error(self):
        with self.assertRaises(ValueError):
            ornith_config_from_hf({"hidden_size": 16})

    def test_convert_hf_produces_runnable_checkpoint(self):
        dsq_path = os.path.join(self.tmpdir.name, "from_hf.dsq")
        convert_hf_to_dspark(self.hf_dir, dsq_path, quant="int8", block_size=8)

        with DsparkCheckpoint(dsq_path) as ckpt:
            self.assertEqual(ckpt.model_family, "ornith")
            model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=1)
            out = model.generate(prompt_ids=[1, 2, 3], max_new_tokens=4)
            self.assertEqual(len(out), 4)

    def test_convert_hf_matches_direct_source_conversion(self):
        # The safetensors fixture and the "source" fixture below both encode
        # the exact same underlying float values, just through different
        # container formats - so quantizing either should produce bit-identical
        # .dsq output (same values, same algorithm, same block size).
        source_path = os.path.join(self.tmpdir.name, "source.bin")
        write_source_checkpoint(source_path, self.config.to_dict(), self.tensors)
        direct_dsq = os.path.join(self.tmpdir.name, "direct.dsq")
        convert_source_to_dspark(source_path, direct_dsq, quant="int8", block_size=8)

        hf_dsq = os.path.join(self.tmpdir.name, "hf.dsq")
        convert_hf_to_dspark(self.hf_dir, hf_dsq, quant="int8", block_size=8)

        with DsparkCheckpoint(direct_dsq) as direct_ckpt, DsparkCheckpoint(hf_dsq) as hf_ckpt:
            self.assertEqual(sorted(direct_ckpt.tensor_names()), sorted(hf_ckpt.tensor_names()))
            for name in direct_ckpt.tensor_names():
                direct_values = direct_ckpt.get_tensor(name).load()
                hf_values = hf_ckpt.get_tensor(name).load()
                self.assertEqual(direct_values, hf_values)

    def test_missing_tensor_raises_clear_error(self):
        bad_map = dict(DEFAULT_HF_NAME_MAP)
        bad_map["embed_tokens.weight"] = "does.not.exist"
        dsq_path = os.path.join(self.tmpdir.name, "bad.dsq")
        with self.assertRaises(KeyError):
            convert_hf_to_dspark(self.hf_dir, dsq_path, name_map=bad_map)


if __name__ == "__main__":
    unittest.main()
