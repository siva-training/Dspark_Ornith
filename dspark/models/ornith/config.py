from dataclasses import dataclass

from dspark.base import BaseDsparkConfig


@dataclass
class OrnithConfig(BaseDsparkConfig):
    """Configuration for the Ornith decoder-only transformer family.

    Ornith uses the same architecture dspark's core primitives implement
    (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU) - it is
    "from a similar family" to what dspark already targets, so no new math
    is needed in :mod:`dspark.nn`, only Ornith-specific config defaults and
    checkpoint I/O.
    """

    model_type: str = "ornith"


#: Convenience size presets used by the CLI/tests/benchmarks to synthesize
#: small-but-representative Ornith checkpoints without needing real
#: pretrained weights.
PRESETS = {
    "tiny": dict(
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=256,
        intermediate_size=172,
        max_position_embeddings=256,
    ),
    "small": dict(
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        vocab_size=256,
        intermediate_size=688,
        max_position_embeddings=512,
    ),
}


def preset_config(name):
    if name not in PRESETS:
        raise KeyError(f"unknown Ornith preset {name!r}. Options: {sorted(PRESETS)}")
    return OrnithConfig(**PRESETS[name])
