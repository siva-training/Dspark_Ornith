import math
import random
import unittest

from dspark.quant import (
    dequantize_blockwise_int4,
    dequantize_blockwise_int8,
    quantize_blockwise_int4,
    quantize_blockwise_int8,
)


class TestBlockwiseInt8(unittest.TestCase):
    def test_round_trip_within_tolerance(self):
        rng = random.Random(42)
        values = [rng.uniform(-3.0, 3.0) for _ in range(257)]
        data, scales = quantize_blockwise_int8(values, block_size=64)
        restored = dequantize_blockwise_int8(data, scales, block_size=64, count=len(values))
        self.assertEqual(len(restored), len(values))
        for original, got in zip(values, restored):
            self.assertLess(abs(original - got), 0.03)

    def test_all_zero_block(self):
        data, scales = quantize_blockwise_int8([0.0] * 10, block_size=4)
        restored = dequantize_blockwise_int8(data, scales, block_size=4, count=10)
        self.assertEqual(restored, [0.0] * 10)

    def test_clips_outliers(self):
        # Two values in the same block with very different magnitudes: the
        # smaller one is quantized relative to the block's max, so it will
        # lose precision, but the round trip must not raise or overflow.
        data, scales = quantize_blockwise_int8([100.0, 0.0001], block_size=2)
        restored = dequantize_blockwise_int8(data, scales, block_size=2, count=2)
        self.assertAlmostEqual(restored[0], 100.0, delta=1.0)


class TestBlockwiseInt4(unittest.TestCase):
    def test_round_trip_within_tolerance(self):
        rng = random.Random(7)
        values = [rng.uniform(-1.0, 1.0) for _ in range(65)]
        data, scales = quantize_blockwise_int4(values, block_size=32)
        restored = dequantize_blockwise_int4(data, scales, count=len(values), block_size=32)
        self.assertEqual(len(restored), len(values))
        for original, got in zip(values, restored):
            self.assertLess(abs(original - got), 0.2)

    def test_odd_length_packing(self):
        values = [0.5, -0.25, 0.75]
        data, scales = quantize_blockwise_int4(values, block_size=3)
        self.assertEqual(len(data), 2)  # ceil(3/2)
        restored = dequantize_blockwise_int4(data, scales, count=3, block_size=3)
        self.assertEqual(len(restored), 3)
        for original, got in zip(values, restored):
            self.assertLess(abs(original - got), 0.15)

    def test_smaller_than_int8_on_disk(self):
        rng = random.Random(1)
        values = [rng.uniform(-1, 1) for _ in range(128)]
        int8_data, _ = quantize_blockwise_int8(values, block_size=32)
        int4_data, _ = quantize_blockwise_int4(values, block_size=32)
        self.assertLess(len(int4_data), len(int8_data))


if __name__ == "__main__":
    unittest.main()
