import os
import random
import tempfile
import unittest

from dspark.quant import quantize_blockwise_int4, quantize_blockwise_int8
from dspark.storage import DsparkCheckpoint, DsparkWriter


class TestDsparkStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write_checkpoint(self, path):
        rng = random.Random(0)
        matrix = [rng.uniform(-1, 1) for _ in range(4 * 8)]  # shape (4, 8)
        vector = [rng.uniform(-1, 1) for _ in range(16)]

        writer = DsparkWriter(model_family="ornith", config={"hidden_size": 8, "num_layers": 1})
        data8, scales8 = quantize_blockwise_int8(matrix, block_size=8)  # one block per row
        writer.add_tensor("rows.weight", "int8", [4, 8], 8, data8, scales8)
        data4, scales4 = quantize_blockwise_int4(vector, block_size=16)
        writer.add_tensor("flat.weight", "int4", [16], 16, data4, scales4)
        writer.save(path)
        return matrix, vector

    def test_round_trip_load(self):
        path = os.path.join(self.tmpdir.name, "test.dsq")
        matrix, vector = self._write_checkpoint(path)

        with DsparkCheckpoint(path) as ckpt:
            self.assertEqual(ckpt.model_family, "ornith")
            self.assertEqual(sorted(ckpt.tensor_names()), ["flat.weight", "rows.weight"])

            rows_tensor = ckpt.get_tensor("rows.weight")
            restored = rows_tensor.load()
            self.assertEqual(len(restored), len(matrix))
            for original, got in zip(matrix, restored):
                self.assertLess(abs(original - got), 0.03)

            flat_tensor = ckpt.get_tensor("flat.weight")
            restored_flat = flat_tensor.load()
            for original, got in zip(vector, restored_flat):
                self.assertLess(abs(original - got), 0.2)

    def test_get_row_matches_full_load(self):
        path = os.path.join(self.tmpdir.name, "test.dsq")
        self._write_checkpoint(path)

        with DsparkCheckpoint(path) as ckpt:
            tensor = ckpt.get_tensor("rows.weight")
            full = tensor.load()
            for row_idx in range(4):
                row = tensor.get_row(row_idx)
                expected = full[row_idx * 8 : (row_idx + 1) * 8]
                for a, b in zip(row, expected):
                    self.assertAlmostEqual(a, b, places=5)

    def test_get_row_requires_one_block_per_row(self):
        path = os.path.join(self.tmpdir.name, "bad.dsq")
        writer = DsparkWriter(model_family="ornith", config={})
        data, scales = quantize_blockwise_int8([0.0] * 16, block_size=4)  # 4 blocks, not 1
        writer.add_tensor("bad.weight", "int8", [2, 8], 4, data, scales)
        writer.save(path)

        with DsparkCheckpoint(path) as ckpt:
            with self.assertRaises(ValueError):
                ckpt.get_tensor("bad.weight").get_row(0)

    def test_unknown_tensor_raises_key_error(self):
        path = os.path.join(self.tmpdir.name, "test.dsq")
        self._write_checkpoint(path)
        with DsparkCheckpoint(path) as ckpt:
            with self.assertRaises(KeyError):
                ckpt.get_tensor("does.not.exist")

    def test_bad_magic_raises(self):
        path = os.path.join(self.tmpdir.name, "notdspark.bin")
        with open(path, "wb") as f:
            f.write(b"not a dspark file at all")
        with self.assertRaises(ValueError):
            DsparkCheckpoint(path)


if __name__ == "__main__":
    unittest.main()
