import math
import os
import tempfile
import unittest

from dspark.models.ornith.baseline import NaiveOrnithModel
from dspark.models.ornith.config import OrnithConfig, preset_config
from dspark.models.ornith.convert import convert_source_to_dspark
from dspark.models.ornith.model import OrnithForCausalLM
from dspark.models.ornith.synthetic import write_synthetic_ornith_checkpoint
from dspark.models.ornith.tokenizer import ByteTokenizer
from dspark.storage import DsparkCheckpoint


class TestOrnithEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.config = OrnithConfig(
            hidden_size=16,
            num_layers=3,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=32,
            intermediate_size=24,
            max_position_embeddings=64,
        )
        self.source_path = os.path.join(self.tmpdir.name, "source.bin")
        write_synthetic_ornith_checkpoint(self.source_path, self.config, seed=123)
        self.dsq_path = os.path.join(self.tmpdir.name, "model.dsq")
        convert_source_to_dspark(self.source_path, self.dsq_path, quant="int8", block_size=8)

    def test_generate_produces_requested_token_count(self):
        with DsparkCheckpoint(self.dsq_path) as ckpt:
            model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=1)
            out = model.generate(prompt_ids=[1, 2, 3], max_new_tokens=5)
            self.assertEqual(len(out), 5)
            for token_id in out:
                self.assertTrue(0 <= token_id < self.config.vocab_size)

    def test_layer_cache_respects_max_resident_layers(self):
        with DsparkCheckpoint(self.dsq_path) as ckpt:
            model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=2)
            model.generate(prompt_ids=[0, 1, 2, 3], max_new_tokens=0)
            self.assertLessEqual(len(model.resident_layer_indices()), 2)

    def test_int4_checkpoint_also_runs(self):
        dsq4_path = os.path.join(self.tmpdir.name, "model_int4.dsq")
        convert_source_to_dspark(self.source_path, dsq4_path, quant="int4", block_size=8)
        with DsparkCheckpoint(dsq4_path) as ckpt:
            model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=1)
            out = model.generate(prompt_ids=[4, 5], max_new_tokens=3)
            self.assertEqual(len(out), 3)

    def test_quantized_generation_is_deterministic(self):
        with DsparkCheckpoint(self.dsq_path) as ckpt1:
            m1 = OrnithForCausalLM.from_checkpoint(ckpt1, max_resident_layers=1)
            out1 = m1.generate([1, 2, 3], max_new_tokens=4)
        with DsparkCheckpoint(self.dsq_path) as ckpt2:
            m2 = OrnithForCausalLM.from_checkpoint(ckpt2, max_resident_layers=3)
            out2 = m2.generate([1, 2, 3], max_new_tokens=4)
        # max_resident_layers only affects what is cached, never the math.
        self.assertEqual(out1, out2)

    def test_quantized_close_to_naive_fp32_baseline(self):
        naive = NaiveOrnithModel.from_source(self.source_path)
        naive_logits = naive.step(token_id=2, position=0)

        with DsparkCheckpoint(self.dsq_path) as ckpt:
            model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=1)
            dspark_logits = model.step(token_id=2, position=0)

        self.assertEqual(len(naive_logits), len(dspark_logits))
        # int8 quantization introduces small error; the argmax over such a
        # small, randomly initialized model may legitimately differ, so we
        # only assert the logits are numerically close, not identical.
        max_abs_logit = max(abs(v) for v in naive_logits)
        for a, b in zip(naive_logits, dspark_logits):
            self.assertLess(abs(a - b), 0.5 + 0.1 * max_abs_logit)

    def test_rejects_wrong_family_checkpoint(self):
        from dspark.storage import DsparkWriter

        bad_path = os.path.join(self.tmpdir.name, "bad.dsq")
        DsparkWriter(model_family="not-ornith", config={}).save(bad_path)
        with DsparkCheckpoint(bad_path) as ckpt:
            with self.assertRaises(ValueError):
                OrnithForCausalLM.from_checkpoint(ckpt)


class TestPresetsAndTokenizer(unittest.TestCase):
    def test_presets_are_constructible(self):
        for name in ("tiny", "small"):
            config = preset_config(name)
            self.assertGreater(config.num_layers, 0)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            preset_config("does-not-exist")

    def test_byte_tokenizer_round_trip(self):
        tok = ByteTokenizer()
        text = "hello, dspark!"
        ids = tok.encode(text)
        self.assertTrue(all(0 <= i < 256 for i in ids))
        self.assertEqual(tok.decode(ids), text)


if __name__ == "__main__":
    unittest.main()
