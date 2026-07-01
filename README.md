# dspark

dspark is a small, dependency-free inference runtime built to run models
from the Ornith family on consumer hardware, with a peak memory footprint
far below what naively loading the checkpoint would require.

It has no numpy/torch dependency at all - everything is Python's standard
library. That is itself part of the "lesser memory footprint" story: dspark
adds zero framework overhead on top of the techniques it uses to shrink the
model itself.

## Why Ornith fits dspark

Ornith is a decoder-only transformer (RMSNorm, rotary position embeddings,
grouped-query attention, SwiGLU MLP) - the same architecture family dspark's
core (`dspark/nn.py`) already implements. Because of that, enabling dspark
for Ornith did not require new attention/MLP math: it only required Ornith's
config (`dspark/models/ornith/config.py`) and checkpoint I/O
(`dspark/models/ornith/{source,convert,synthetic}.py`), wired into dspark's
runtime (`dspark/models/ornith/model.py`) and its model-family registry
(`dspark/registry.py`). Any other model from the same family could plug in
the same way without touching dspark's core.

## How the memory footprint is reduced

Three techniques compose to bound peak RAM independent of total model size:

1. **Blockwise int8 / int4 quantization** (`dspark/quant.py`) - weights are
   stored as 1-byte or 4-bit integers with a small per-block float32 scale,
   roughly a 4x-8x reduction over float32 on disk.
2. **Memory-mapped checkpoints** (`dspark/storage.py`, the `.dsq` format) -
   the quantized checkpoint is opened with `mmap`, so weight bytes live in
   OS page cache and are paged in lazily, not read wholesale into the
   process heap. The token embedding / `lm_head` tables (often the largest
   tensors in a model, scaling with vocabulary size) are further quantized
   one block per row so a single token lookup decodes only that one row.
3. **Bounded layer streaming** (`dspark/models/ornith/model.py`) - decoder
   layers are decoded into Python objects lazily and kept in an LRU cache
   of size `max_resident_layers`. Once that many layers are resident, the
   least-recently-used one is evicted before the next is decoded, so a
   model with far more layers than fit comfortably in RAM still runs to
   completion, at the cost of re-decoding evicted layers on the next pass.

## Layout

```
dspark/
  quant.py            blockwise int8/int4 quantize + dequantize
  storage.py           .dsq container format, mmap-backed lazy tensor access
  nn.py                 shared decoder-only transformer math (the "family")
  base.py               BaseDsparkConfig / BaseDsparkModel contracts
  registry.py            pluggable model-family registry
  memory_utils.py         peak RSS helpers used by the benchmark
  cli.py                    `python -m dspark ...`
  models/ornith/
    config.py            OrnithConfig + tiny/small presets
    source.py             uncompressed fp32 "source" checkpoint format
    synthetic.py           deterministic synthetic Ornith checkpoints (no
                            pretrained Ornith weights are available here)
    convert.py              source -> quantized .dsq conversion
    model.py                 OrnithForCausalLM: mmap + quantized + streamed
    baseline.py               naive fully-resident fp32 model, for comparison
    tokenizer.py               minimal byte-level tokenizer for demos
tests/                          unittest suite
```

## CLI

```bash
# 1. Synthesize a small Ornith checkpoint (stand-in for a real training export)
python -m dspark synth-ornith --out ornith_source.bin --preset small

# 2. Quantize it into dspark's memory-mapped format
python -m dspark convert --src ornith_source.bin --out ornith.dsq --quant int4

# 3. Run it with a bounded number of resident layers
python -m dspark run --checkpoint ornith.dsq --prompt "hello" \
    --max-new-tokens 32 --max-resident-layers 2

# 4. Inspect its on-disk / resident footprint
python -m dspark inspect --checkpoint ornith.dsq --max-resident-layers 1 2 4

# 5. Compare peak RSS against a naive, fully-resident fp32 run
python -m dspark benchmark --preset small --quant int4 --max-resident-layers 2
```

`benchmark` synthesizes a checkpoint, converts it, runs both a naive
fully-resident fp32 model and dspark's streamed/quantized model each in
their own subprocess, and reports each one's peak resident set size (RSS)
along with the reduction factor.

## Running the tests

```bash
python -m unittest discover -s tests -v
```
