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
    hf_import.py             Hugging Face (config.json + .safetensors) -> .dsq
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

## Running a real Ornith checkpoint downloaded from Hugging Face

`convert-hf` reads directly from a Hugging Face checkpoint directory - a
`config.json` plus one or more `.safetensors` files - and quantizes straight
into dspark's `.dsq` format. It needs neither `torch`, `transformers`, nor
the `safetensors` pip package; it parses the (simple, documented) safetensors
container itself with the standard library.

```bash
# 1. Download the checkpoint (only config.json + *.safetensors are needed;
#    skip tokenizer files, .bin/.pt weights, etc.)
huggingface-cli download <org>/<ornith-model> \
    --local-dir ./ornith-hf --include "*.json" "*.safetensors"

# 2. Quantize it directly into dspark's format - no fp32 intermediate file,
#    and no full-precision copy of the model is ever held in memory at once
python -m dspark convert-hf --hf-dir ./ornith-hf --out ornith.dsq --quant int4

# 3. Run it exactly like any other .dsq checkpoint
python -m dspark run --checkpoint ornith.dsq --prompt "hello" \
    --max-new-tokens 64 --max-resident-layers 2
```

If the checkpoint is sharded (`model-00001-of-0000N.safetensors` +
`model.safetensors.index.json`), `convert-hf` follows the index
automatically - just make sure all the shard files are downloaded alongside it.

`convert-hf` assumes the checkpoint's tensors are named the way
LLaMA/Mistral-family checkpoints on Hugging Face conventionally are
(`model.embed_tokens.weight`, `model.layers.<i>.self_attn.q_proj.weight`,
...) and that `config.json` uses the matching field names
(`num_hidden_layers`, `num_attention_heads`, ...) - see
`dspark/models/ornith/hf_import.py` for the exact mapping
(`DEFAULT_HF_NAME_MAP` / `HF_CONFIG_FIELD_MAP`). If a specific Ornith
release uses different tensor or config field names:

- `convert_hf_to_dspark(..., name_map=my_custom_map)` accepts a
  replacement tensor name mapping.
- Missing config fields or tensors raise a clear error naming exactly
  what was expected and (for tensors) a sample of what was actually found,
  rather than failing silently or with a shape mismatch deep in the model.

Only the `.safetensors` format is supported, not `pytorch_model.bin` -
safetensors doesn't require executing arbitrary pickled code to load,
which matters for checkpoints pulled from the internet, and its simple
header+blob layout is what makes tensor-at-a-time, low-memory conversion
possible in the first place. If a checkpoint only ships `.bin`/`.pt`
weights, convert it to safetensors first (the `safetensors` library's
`convert.py` / the Hugging Face Hub's "safetensors" conversion PR bot can
do this) before pointing `convert-hf` at it.

## Sizing hardware (including GCP) for a given Ornith checkpoint

See [`docs/gcp-sizing.md`](docs/gcp-sizing.md) for RAM/disk/vCPU sizing
guidance across Ornith model sizes, including concrete GCP machine-type
recommendations and the memory model (checkpoint size, resident-layer
footprint, KV cache) they're derived from.

## Running the tests

```bash
python -m unittest discover -s tests -v
```
