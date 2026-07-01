# Sizing GCP machines for Ornith on dspark

This is a sizing guide for running Ornith-family checkpoints through dspark
on Google Compute Engine. It covers RAM, disk, and vCPU, and is explicit
about which numbers are exact (checkpoint size on disk) versus estimated
(peak process RAM), so you can tell how much margin to leave.

**Bottom line up front:** run `dspark inspect` (and, where feasible,
`dspark benchmark`) against your *actual* checkpoint before committing to a
machine size in production. The tables below are reference points derived
from representative configs and a memory model measured on dspark's small
built-in presets - they are a starting point for provisioning, not a
substitute for measuring your specific model.

## Why there are two different kinds of numbers here

- **Checkpoint size on disk** (fp32 / int8 / int4) is computed directly from
  `dspark.quant.quantized_size_bytes` - exact, not an estimate, for any
  config you plug in.
- **Peak process RAM** cannot be stated exactly without running the model,
  for one specific reason: dspark's pure-Python runtime (see the main
  [README](../README.md)) stores decoded weights as Python float objects,
  not packed native arrays. A boxed Python float plus its list slot costs
  roughly 10x a packed float32 (measured below), and that overhead - not
  the quantization ratio - ends up dominating resident memory. The RAM
  estimates in this doc bake that overhead in; see "Where the numbers come
  from" for exactly how.

## Reference Ornith configurations used below

Ornith's own published configs weren't available to size against directly,
so each tier below uses a config representative of a well-known open
decoder-only transformer of that parameter count (same architecture family:
RMSNorm, RoPE, grouped-query attention, SwiGLU). If you have Ornith's actual
`config.json`, use it directly - see "Sizing your exact checkpoint" below.

| Tier | hidden | layers | heads | kv heads | intermediate | vocab | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ornith-1B  | 2048 | 22 | 32 | 4  | 5632  | 32000 | 1.03B |
| Ornith-3B  | 3072 | 26 | 24 | 8  | 8192  | 32000 | 2.72B |
| Ornith-8B  | 4096 | 32 | 32 | 8  | 14336 | 32000 | 7.11B |
| Ornith-13B | 5120 | 40 | 40 | 40 | 13824 | 32000 | 12.85B |
| Ornith-70B | 8192 | 80 | 64 | 8  | 28672 | 32000 | 68.71B |

Note Ornith-13B here (mirroring Llama-2-13B) uses plain multi-head attention
(`num_kv_heads == num_heads`, no GQA), so its KV cache scales much worse
with context length than the 8B/70B tiers, which use grouped-query
attention (8 kv heads). If the real Ornith-13B analog uses GQA, its KV
cache numbers below will be pessimistic - swap in the real `config.json` to
know for sure.

## Checkpoint size on disk (exact)

| Tier | fp32 (uncompressed source) | int8 `.dsq` | int4 `.dsq` |
|---|---:|---:|---:|
| Ornith-1B  | 3.85 GiB | 1.02 GiB | 0.54 GiB |
| Ornith-3B  | 10.12 GiB | 2.69 GiB | 1.42 GiB |
| Ornith-8B  | 26.49 GiB | 7.04 GiB | 3.73 GiB |
| Ornith-13B | 47.88 GiB | 12.72 GiB | 6.73 GiB |
| Ornith-70B | 255.98 GiB | 68.00 GiB | 36.00 GiB |

(fp32 column is dspark's own "source" checkpoint format, i.e. what you'd get
before quantizing; if you're converting straight from Hugging Face
safetensors via `convert-hf`, you never materialize this file at all.)

## Where the peak-RAM numbers come from

Three components add up to a running `dspark run`/`convert-hf` process's
peak RSS:

1. **Base process overhead** - the Python interpreter plus dspark's own
   imports (~200 MiB, generous).
2. **Resident layer weights** - `max_resident_layers` x (per-layer
   float32-equivalent size) x **~10.3**, the measured Python object-overhead
   multiplier (see below). This is the number `--max-resident-layers`
   directly controls.
3. **KV cache** - unlike layer weights, the KV cache is **not** bounded by
   `max_resident_layers`: it holds one entry per layer x per generated
   position x per KV head, for the *entire* model, for as long as the
   sequence is live. It scales with `num_layers x context_length x kv_dim`,
   with the same ~10.3x Python object overhead applied.

The 10.3x multiplier is measured, not assumed: `dspark benchmark --preset
small` (hidden_size=256, 8 layers) reports peak RSS of 76.59 MiB at
`max_resident_layers=1` and 158.07 MiB at `max_resident_layers=4`. The
marginal cost per additional resident layer is `(158.07-76.59)/3 = 27.16
MiB`, against a `per-layer` float32-equivalent size of 2.64 MiB - a 10.3x
ratio. Because this overhead is a fixed per-Python-float-object cost (not
something that depends on matrix shape), it's reasonable to extrapolate
across model sizes, but it hasn't been directly re-verified at 8B+ scale in
this environment (synthesizing billions of test floats in pure Python isn't
practical) - treat the larger tiers' numbers as directionally right rather
than exact.

## Estimated peak RAM by tier

Assuming int4 quantization and a 4096-token context (KV cache scales
linearly with context length - halve the KV-cache column for a 2048-token
context, etc.):

| Tier | max_resident_layers=1 | max_resident_layers=2 | max_resident_layers=4 |
|---|---:|---:|---:|
| Ornith-1B  | 3.7 GiB | 5.4 GiB | 8.7 GiB |
| Ornith-3B  | 12.4 GiB | 16.3 GiB | 24.0 GiB |
| Ornith-8B  | 18.9 GiB | 27.2 GiB | 44.0 GiB |
| Ornith-13B | 76.7 GiB | 88.9 GiB | 113.3 GiB |
| Ornith-70B | 58.8 GiB | 91.6 GiB | 157.3 GiB |

Ornith-13B's KV cache (no GQA, see above) dominates its total here; its
resident-layer contribution alone is actually smaller than Ornith-8B's.

**This is the central finding of this doc:** dspark's quantization and
layer-streaming give a large, *measured* reduction relative to a naive
fully-resident fp32 load (2.4x-3.2x, see the main README) - but because the
runtime is pure Python with no native array backend, that reduction doesn't
carry all the way down to "runs in a few GB" at 13B+ scale the way a
native/vectorized quantized runtime (e.g. llama.cpp) would. dspark's
sweet spot for genuinely small-footprint deployment is the **1B-8B range**;
above that, you're still provisioning a real high-memory VM, just a
meaningfully smaller one than a naive fp32 load would need, with a much
smaller on-disk footprint.

## Recommended GCP machine types

Default assumption: `max_resident_layers=2`, int4 quantization, ~4k
context. Sized with roughly 30-40% headroom over the estimate above for OS
page cache, other processes, and safety margin. "Minimum" trims that margin
for cost-sensitive/dev use; "Recommended" is what to actually run in
production.

| Tier | Minimum | Recommended | Boot/checkpoint disk |
|---|---|---|---|
| Ornith-1B  | e2-standard-2 (2 vCPU, 8 GB) | e2-standard-4 (4 vCPU, 16 GB) | 20 GB pd-balanced |
| Ornith-3B  | e2-standard-4 (4 vCPU, 16 GB) | e2-standard-8 (8 vCPU, 32 GB) | 30 GB pd-balanced |
| Ornith-8B  | n2-standard-8 (8 vCPU, 32 GB) | n2-highmem-8 (8 vCPU, 64 GB) | 50 GB pd-ssd |
| Ornith-13B | n2-highmem-8 (8 vCPU, 64 GB) | n2-highmem-16 (16 vCPU, 128 GB) | 80 GB pd-ssd |
| Ornith-70B | n2-highmem-16 (16 vCPU, 128 GB)* | n2-highmem-32 (32 vCPU, 256 GB) | 375 GB Local SSD |

\* Ornith-70B at `max_resident_layers=1` (58.8 GiB estimate) is the only
combination in this doc where the 128 GB "minimum" tier isn't just cost
trimming - it's close enough to the estimate that you should verify with a
real run before relying on it, and prefer the 256 GB recommendation for any
non-trivial context length.

If you don't need GPU-accelerated inference (dspark has no GPU path - see
the performance caveat below), none of these need an attached accelerator;
this is pure CPU/RAM/disk sizing.

## Disk: why it matters more than it looks

When `max_resident_layers < num_layers` (true for every recommendation
above except the smallest models at high `max_resident_layers`), the LRU
layer cache evicts and re-decodes layers on essentially every forward pass
through the model - i.e. dspark re-reads most of the checkpoint from disk
(via `mmap`) on **every generated token**, not just once at load time. Disk
random-read throughput directly limits generation speed once this happens.

- For Ornith-8B and below: `pd-ssd` (SSD persistent disk) is sufficient -
  cheap, persistent across restarts, decent random-read IOPS.
- For Ornith-13B/70B, or any deployment pushing `max_resident_layers` down
  to control RAM: attach a **Local SSD** (375 GB per device, stripe
  multiple devices for the largest checkpoints) and copy the `.dsq` file
  onto it at instance startup. Local SSD is ephemeral (wiped on stop), so
  keep the source-of-truth checkpoint on `pd-standard`/GCS and treat the
  Local SSD copy as a cache.
- Avoid `pd-standard` (spinning-disk-class persistent disk) for anything
  above Ornith-1B - its IOPS ceiling will bottleneck generation speed long
  before RAM does.

## Performance caveat: this is a memory-footprint optimizer, not a speed one

dspark's core loop (`dspark/nn.py`) is plain Python arithmetic over lists -
there is no numpy/BLAS/vectorization and no GPU path. It trades inference
*speed* for memory *footprint* and dependency-free portability. Expect
noticeably slower tokens/sec than a native runtime (llama.cpp, vLLM, etc.)
at every tier in this doc, and expect that gap to widen with model size,
since larger hidden/intermediate dimensions mean more pure-Python
inner-product loops per token. Extra vCPUs mostly help you serve more
*concurrent* requests, not make a single request faster (generation is
single-threaded per request). If low-latency serving matters more than
memory footprint or dependency-free deployment for your use case, dspark
in its current form is not the right tool - it's aimed at the case where
minimizing RAM/disk/dependencies is the binding constraint (e.g. genuinely
memory-constrained consumer hardware), not at maximizing throughput.

## Sizing your exact checkpoint

Don't extrapolate from this doc's reference tiers if you can measure
directly. Once you have Ornith's real `config.json`:

```bash
# Exact on-disk size for your quantization choice, and a rough per-layer
# residency estimate:
python -m dspark convert-hf --hf-dir ./ornith-hf --out ornith.dsq --quant int4
python -m dspark inspect --checkpoint ornith.dsq --max-resident-layers 1 2 4

# If the model is small enough to run in this environment, measure real
# peak RSS directly instead of estimating it:
python -m dspark benchmark --preset small --quant int4 --max-resident-layers 2
```

`dspark benchmark` only ships with the built-in `tiny`/`small` synthetic
presets today; to get a real measured number for your actual checkpoint,
run `dspark run` under `/usr/bin/time -v` (Linux) or a memory profiler on
the target machine itself and compare against this doc's estimate.
