"""Deterministic synthetic Ornith checkpoints, for tests, demos, and the
memory benchmark - dspark's converter needs *some* Ornith weights to
exercise, and no real pretrained Ornith checkpoint is available here.
Weights are small random floats; this is for exercising the pipeline
(shapes, quantization, streaming, generation), not producing coherent text.
"""

import random

from .source import write_source_checkpoint

LAYERNORM_INIT = 1.0


def _rand_matrix(rng, out_features, in_features, scale=0.02):
    return [rng.uniform(-scale, scale) for _ in range(out_features * in_features)]


def _const_vec(n, value=LAYERNORM_INIT):
    return [value] * n


def generate_synthetic_ornith_tensors(config, seed=0):
    rng = random.Random(seed)
    h = config.hidden_size
    head_dim = config.head_dim
    kv_dim = head_dim * config.num_kv_heads
    inter = config.intermediate_size
    vocab = config.vocab_size

    tensors = {"embed_tokens.weight": ((vocab, h), _rand_matrix(rng, vocab, h))}
    for layer in range(config.num_layers):
        p = f"layers.{layer}."
        tensors[p + "input_layernorm.weight"] = ((h,), _const_vec(h))
        tensors[p + "self_attn.q_proj.weight"] = ((h, h), _rand_matrix(rng, h, h))
        tensors[p + "self_attn.k_proj.weight"] = ((kv_dim, h), _rand_matrix(rng, kv_dim, h))
        tensors[p + "self_attn.v_proj.weight"] = ((kv_dim, h), _rand_matrix(rng, kv_dim, h))
        tensors[p + "self_attn.o_proj.weight"] = ((h, h), _rand_matrix(rng, h, h))
        tensors[p + "post_attention_layernorm.weight"] = ((h,), _const_vec(h))
        tensors[p + "mlp.gate_proj.weight"] = ((inter, h), _rand_matrix(rng, inter, h))
        tensors[p + "mlp.up_proj.weight"] = ((inter, h), _rand_matrix(rng, inter, h))
        tensors[p + "mlp.down_proj.weight"] = ((h, inter), _rand_matrix(rng, h, inter))
    tensors["norm.weight"] = ((h,), _const_vec(h))
    if not config.tie_word_embeddings:
        tensors["lm_head.weight"] = ((vocab, h), _rand_matrix(rng, vocab, h))
    return tensors


def write_synthetic_ornith_checkpoint(path, config, seed=0):
    tensors = generate_synthetic_ornith_tensors(config, seed=seed)
    write_source_checkpoint(path, config.to_dict(), tensors)
