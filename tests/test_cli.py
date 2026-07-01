import json
import os
import subprocess
import sys
import tempfile
import unittest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "dspark", *args],
        capture_output=True,
        text=True,
        check=True,
    )


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_synth_convert_run_inspect(self):
        source_path = os.path.join(self.tmpdir.name, "source.bin")
        dsq_path = os.path.join(self.tmpdir.name, "model.dsq")

        run_cli("synth-ornith", "--out", source_path, "--preset", "tiny", "--seed", "1")
        self.assertTrue(os.path.exists(source_path))

        run_cli("convert", "--src", source_path, "--out", dsq_path, "--family", "ornith", "--quant", "int4")
        self.assertTrue(os.path.exists(dsq_path))

        run_result = run_cli(
            "run", "--checkpoint", dsq_path, "--prompt", "hi", "--max-new-tokens", "4",
            "--max-resident-layers", "1",
        )
        self.assertTrue(len(run_result.stdout) > 0)

        inspect_result = run_cli("inspect", "--checkpoint", dsq_path, "--max-resident-layers", "1", "2")
        self.assertIn("family:", inspect_result.stdout)
        self.assertIn("ornith", inspect_result.stdout)

    def test_convert_hf_end_to_end(self):
        from tests.test_hf_import import dspark_template_to_hf_name, write_safetensors

        from dspark.models.ornith.config import preset_config
        from dspark.models.ornith.synthetic import generate_synthetic_ornith_tensors

        config = preset_config("tiny")
        tensors = generate_synthetic_ornith_tensors(config, seed=5)

        hf_dir = os.path.join(self.tmpdir.name, "hf_ornith")
        os.makedirs(hf_dir)
        with open(os.path.join(hf_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "hidden_size": config.hidden_size,
                    "num_hidden_layers": config.num_layers,
                    "num_attention_heads": config.num_heads,
                    "num_key_value_heads": config.num_kv_heads,
                    "vocab_size": config.vocab_size,
                    "intermediate_size": config.intermediate_size,
                    "max_position_embeddings": config.max_position_embeddings,
                },
                f,
            )
        hf_tensors = {}
        for name, value in tensors.items():
            if name.startswith("layers."):
                layer = int(name.split(".")[1])
                rest = name.split(".", 2)[2]
                hf_name = dspark_template_to_hf_name(f"layers.{{i}}.{rest}", layer=layer)
            else:
                hf_name = dspark_template_to_hf_name(name)
            hf_tensors[hf_name] = value
        write_safetensors(os.path.join(hf_dir, "model.safetensors"), hf_tensors)

        dsq_path = os.path.join(self.tmpdir.name, "from_hf.dsq")
        run_cli("convert-hf", "--hf-dir", hf_dir, "--out", dsq_path, "--quant", "int4")
        self.assertTrue(os.path.exists(dsq_path))

        run_result = run_cli(
            "run", "--checkpoint", dsq_path, "--prompt", "hi", "--max-new-tokens", "4",
            "--max-resident-layers", "1",
        )
        self.assertTrue(len(run_result.stdout) > 0)

    def test_benchmark_reports_reduction(self):
        result = run_cli(
            "benchmark", "--preset", "small", "--quant", "int4",
            "--max-resident-layers", "1", "--max-new-tokens", "2",
        )
        self.assertIn("peak RSS reduction", result.stdout)
        reduction_line = next(l for l in result.stdout.splitlines() if "peak RSS reduction" in l)
        factor = float(reduction_line.split(":")[1].strip().rstrip("x"))
        # A real regression (e.g. the benchmark's own driver process ballooning
        # in memory before forking the measured workers) collapses this to ~1.0x,
        # so assert a real, comfortably-detectable reduction rather than just
        # checking the line is present.
        self.assertGreater(factor, 1.5)


if __name__ == "__main__":
    unittest.main()
