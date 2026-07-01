"""Command-line entry point: ``python -m dspark <subcommand>``.

Subcommands:
  synth-ornith   Write a small synthetic Ornith source checkpoint (fp32).
  convert        Quantize a source checkpoint into dspark's .dsq format.
  convert-hf     Quantize a Hugging Face Ornith checkpoint directory
                 (config.json + .safetensors) directly into .dsq.
  run            Load a .dsq checkpoint and greedy-generate text from a prompt.
  inspect        Print size/footprint information about a .dsq checkpoint.
  benchmark      Synthesize + convert an Ornith checkpoint, then compare peak
                 RSS between a naive fully-resident fp32 run and dspark's
                 mmap + quantized + layer-streamed run.
"""

import argparse
import subprocess
import sys

from .memory_utils import peak_rss_mb
from .models.ornith.baseline import NaiveOrnithModel
from .models.ornith.config import PRESETS, preset_config
from .models.ornith.hf_import import convert_hf_to_dspark
from .models.ornith.model import DEFAULT_MAX_RESIDENT_LAYERS, OrnithForCausalLM
from .models.ornith.synthetic import write_synthetic_ornith_checkpoint
from .models.ornith.tokenizer import ByteTokenizer
from .registry import get_model_family, registered_families
from .storage import DsparkCheckpoint


def cmd_synth_ornith(args):
    config = preset_config(args.preset)
    write_synthetic_ornith_checkpoint(args.out, config, seed=args.seed)
    print(f"wrote synthetic Ornith '{args.preset}' source checkpoint to {args.out}")


def cmd_convert(args):
    family = get_model_family(args.family)
    out = family.converter(args.src, args.out, quant=args.quant, block_size=args.block_size)
    print(f"wrote {args.family} .dsq checkpoint ({args.quant}) to {out}")


def cmd_convert_hf(args):
    out = convert_hf_to_dspark(args.hf_dir, args.out, quant=args.quant, block_size=args.block_size)
    print(f"wrote ornith .dsq checkpoint ({args.quant}) to {out}")


def cmd_inspect(args):
    with DsparkCheckpoint(args.checkpoint) as ckpt:
        names = ckpt.tensor_names()
        on_disk_bytes = sum(ckpt.get_tensor(n).on_disk_nbytes() for n in names)
        num_layers = ckpt.config.get("num_layers", 0)
        layer0_names = [n for n in names if n.startswith("layers.0.")]
        float32_equiv_per_layer = sum(ckpt.get_tensor(n).numel() for n in layer0_names) * 4

        print(f"family:          {ckpt.model_family}")
        print(f"tensors:         {len(names)}")
        print(f"file size:       {ckpt.file_size_bytes() / (1024 * 1024):.2f} MiB")
        print(f"quantized data:  {on_disk_bytes / (1024 * 1024):.2f} MiB")
        print(f"layers:          {num_layers}")
        print(
            "per-layer decoded size (float32-equivalent): "
            f"{float32_equiv_per_layer / (1024 * 1024):.2f} MiB"
        )
        for k in args.max_resident_layers:
            resident = k * float32_equiv_per_layer / (1024 * 1024)
            print(f"  max_resident_layers={k}: ~{resident:.2f} MiB of layer weights resident at once")


def cmd_run(args):
    with DsparkCheckpoint(args.checkpoint) as ckpt:
        family = get_model_family(ckpt.model_family)
        model = family.model_cls.from_checkpoint(ckpt, max_resident_layers=args.max_resident_layers)
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(args.prompt)
        generated = model.generate(prompt_ids, args.max_new_tokens)
        print(tokenizer.decode(generated))


def cmd_bench_worker(args):
    """Hidden subcommand run as a fresh subprocess so its own peak RSS can
    be measured in isolation, without also counting the parent CLI process."""
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if args.mode == "naive":
        model = NaiveOrnithModel.from_source(args.source)
    else:
        ckpt = DsparkCheckpoint(args.checkpoint)
        model = OrnithForCausalLM.from_checkpoint(ckpt, max_resident_layers=args.max_resident_layers)
    model.generate(prompt_ids, args.max_new_tokens)
    print(f"PEAK_RSS_MB={peak_rss_mb()}")


def _run_worker_and_measure(cmd):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("PEAK_RSS_MB="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"benchmark worker did not report peak RSS.\nstdout={result.stdout}\nstderr={result.stderr}")


def cmd_benchmark(args):
    import os
    import tempfile

    # Building the synthetic checkpoint and quantizing it can itself use a
    # lot of transient memory (millions of individually-boxed Python float
    # objects). On Linux, fork() gives a freshly-spawned child a *transient*
    # ru_maxrss high-water mark equal to the forking parent's RSS at fork
    # time - even though exec() immediately replaces the child's image with
    # a small process. So if this CLI process did that heavy work in-process
    # and then forked the measurement workers below, both workers' "peak
    # RSS" numbers would be polluted by this process's own memory rather
    # than reflecting the workers' real usage. To keep this process itself
    # small before it forks anything, synth/convert also run as their own
    # subprocesses instead of being called in-process.
    config = preset_config(args.preset)  # cheap: just a dataclass, no tensors

    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "ornith_source.bin")
        dsq_path = os.path.join(tmp, "ornith.dsq")

        subprocess.run(
            [
                sys.executable, "-m", "dspark", "synth-ornith",
                "--out", source_path,
                "--preset", args.preset,
                "--seed", str(args.seed),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "dspark", "convert",
                "--src", source_path,
                "--out", dsq_path,
                "--family", "ornith",
                "--quant", args.quant,
                "--block-size", str(args.block_size),
            ],
            check=True,
            capture_output=True,
        )

        naive_rss = _run_worker_and_measure(
            [
                sys.executable, "-m", "dspark", "_bench-worker",
                "--mode", "naive",
                "--source", source_path,
                "--prompt", args.prompt,
                "--max-new-tokens", str(args.max_new_tokens),
            ]
        )
        dspark_rss = _run_worker_and_measure(
            [
                sys.executable, "-m", "dspark", "_bench-worker",
                "--mode", "dspark",
                "--checkpoint", dsq_path,
                "--prompt", args.prompt,
                "--max-new-tokens", str(args.max_new_tokens),
                "--max-resident-layers", str(args.max_resident_layers),
            ]
        )

        source_size_mb = os.path.getsize(source_path) / (1024 * 1024)
        dsq_size_mb = os.path.getsize(dsq_path) / (1024 * 1024)

        print(f"preset: {args.preset}  (layers={config.num_layers}, hidden_size={config.hidden_size})")
        print(f"source checkpoint (fp32, uncompressed):  {source_size_mb:8.2f} MiB")
        print(f"dspark checkpoint ({args.quant}, quantized): {dsq_size_mb:8.2f} MiB")
        print()
        print(f"naive fully-resident fp32 run   peak RSS: {naive_rss:8.2f} MiB")
        print(
            f"dspark mmap+quantized+streamed  peak RSS: {dspark_rss:8.2f} MiB "
            f"(max_resident_layers={args.max_resident_layers})"
        )
        if dspark_rss > 0:
            print(f"\npeak RSS reduction: {naive_rss / dspark_rss:.2f}x")


def build_parser():
    parser = argparse.ArgumentParser(prog="dspark", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth-ornith", help="write a synthetic Ornith source checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth_ornith)

    p = sub.add_parser("convert", help="quantize a source checkpoint into a .dsq file")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="ornith", choices=registered_families() or ["ornith"])
    p.add_argument("--quant", choices=["int8", "int4"], default="int8")
    p.add_argument("--block-size", type=int, default=64)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser(
        "convert-hf",
        help="quantize a Hugging Face Ornith checkpoint directory (config.json + .safetensors) into .dsq",
    )
    p.add_argument("--hf-dir", required=True, help="directory containing config.json and .safetensors file(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--quant", choices=["int8", "int4"], default="int8")
    p.add_argument("--block-size", type=int, default=64)
    p.set_defaults(func=cmd_convert_hf)

    p = sub.add_parser("run", help="generate text from a .dsq checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-resident-layers", type=int, default=DEFAULT_MAX_RESIDENT_LAYERS)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="print footprint information about a .dsq checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-resident-layers", type=int, nargs="+", default=[1, 2, 4])
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser(
        "benchmark", help="compare peak RSS of a naive fp32 run vs dspark's streamed quantized run"
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--quant", choices=["int8", "int4"], default="int4")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--max-resident-layers", type=int, default=DEFAULT_MAX_RESIDENT_LAYERS)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prompt", default="hello dspark")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("_bench-worker", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=["naive", "dspark"], required=True)
    p.add_argument("--source")
    p.add_argument("--checkpoint")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--max-resident-layers", type=int, default=DEFAULT_MAX_RESIDENT_LAYERS)
    p.set_defaults(func=cmd_bench_worker)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
