"""Shared decoder-only transformer math.

This module is the part of dspark that a whole *family* of architecturally
similar models (RMSNorm + rotary embeddings + grouped-query attention +
SwiGLU MLP, i.e. the LLaMA-style family Ornith belongs to) can reuse as-is.
A family-specific module (e.g. ``dspark.models.ornith``) only needs to own
its config and checkpoint I/O; the actual per-layer math lives here so it
is written, tested, and optimized once.

Everything is plain Python (lists of floats) on purpose: dspark targets
consumer hardware with no guarantee numpy/torch are installed, and pure
Python also makes the memory-footprint story honest (a decoded layer is
exactly the size the Python objects use - nothing hidden in native
buffers).
"""

import math

#: Per-layer weight tensor names a decoder layer needs, relative to a
#: ``layers.<i>.`` prefix. Shared by real (streamed) and baseline models.
DECODER_LAYER_WEIGHT_KEYS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def as_rows(flat, out_features, in_features):
    """View a flat row-major buffer of shape (out_features, in_features)
    as a list of row lists, without copying more than necessary."""
    return [flat[i * in_features : (i + 1) * in_features] for i in range(out_features)]


def matvec(weight_rows, x):
    return [dot(row, x) for row in weight_rows]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def rmsnorm(x, weight, eps):
    mean_sq = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(mean_sq + eps)
    return [(v * inv) * w for v, w in zip(x, weight)]


def silu(v):
    return v / (1.0 + math.exp(-v))


def swiglu(gate, up):
    return [silu(g) * u for g, u in zip(gate, up)]


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def apply_rope(vec, head_dim, position, theta):
    """Rotary position embedding, "rotate-half" convention."""
    half = head_dim // 2
    out = list(vec)
    for i in range(half):
        freq = theta ** (-(2.0 * i) / head_dim)
        angle = position * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x1 = vec[i]
        x2 = vec[i + half]
        out[i] = x1 * cos_a - x2 * sin_a
        out[i + half] = x2 * cos_a + x1 * sin_a
    return out


def argmax(values):
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def new_kv_cache(num_layers):
    return [{"k": [], "v": []} for _ in range(num_layers)]


def transformer_decoder_layer(x, weights, kv_cache_entry, position, cfg):
    """One pre-norm decoder block: self-attention (GQA + RoPE) then SwiGLU
    MLP, each with a residual connection.

    ``weights`` maps the names in :data:`DECODER_LAYER_WEIGHT_KEYS` to
    flat row-major float lists. ``kv_cache_entry`` is one element of the
    list returned by :func:`new_kv_cache` and is mutated in place (a new
    token's key/value are appended).
    """
    head_dim = cfg.hidden_size // cfg.num_heads
    group = cfg.num_heads // cfg.num_kv_heads

    h = rmsnorm(x, weights["input_layernorm.weight"], cfg.rms_norm_eps)

    q_rows = as_rows(weights["self_attn.q_proj.weight"], cfg.num_heads * head_dim, cfg.hidden_size)
    k_rows = as_rows(weights["self_attn.k_proj.weight"], cfg.num_kv_heads * head_dim, cfg.hidden_size)
    v_rows = as_rows(weights["self_attn.v_proj.weight"], cfg.num_kv_heads * head_dim, cfg.hidden_size)
    o_rows = as_rows(weights["self_attn.o_proj.weight"], cfg.hidden_size, cfg.num_heads * head_dim)

    q_full = matvec(q_rows, h)
    k_full = matvec(k_rows, h)
    v_full = matvec(v_rows, h)

    k_heads_new = []
    v_heads_new = []
    for kvh in range(cfg.num_kv_heads):
        k_h = apply_rope(k_full[kvh * head_dim : (kvh + 1) * head_dim], head_dim, position, cfg.rope_theta)
        v_h = v_full[kvh * head_dim : (kvh + 1) * head_dim]
        k_heads_new.append(k_h)
        v_heads_new.append(v_h)
    kv_cache_entry["k"].append(k_heads_new)
    kv_cache_entry["v"].append(v_heads_new)

    scale = 1.0 / math.sqrt(head_dim)
    attn_out = [0.0] * (cfg.num_heads * head_dim)
    for qh in range(cfg.num_heads):
        kvh = qh // group
        q_h = apply_rope(q_full[qh * head_dim : (qh + 1) * head_dim], head_dim, position, cfg.rope_theta)
        scores = [dot(q_h, past[kvh]) * scale for past in kv_cache_entry["k"]]
        attn_weights = softmax(scores)
        acc = [0.0] * head_dim
        for w, past_v in zip(attn_weights, (p[kvh] for p in kv_cache_entry["v"])):
            for i in range(head_dim):
                acc[i] += w * past_v[i]
        attn_out[qh * head_dim : (qh + 1) * head_dim] = acc

    attn_proj = matvec(o_rows, attn_out)
    x = add(x, attn_proj)

    h2 = rmsnorm(x, weights["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    gate_rows = as_rows(weights["mlp.gate_proj.weight"], cfg.intermediate_size, cfg.hidden_size)
    up_rows = as_rows(weights["mlp.up_proj.weight"], cfg.intermediate_size, cfg.hidden_size)
    down_rows = as_rows(weights["mlp.down_proj.weight"], cfg.hidden_size, cfg.intermediate_size)

    gate = matvec(gate_rows, h2)
    up = matvec(up_rows, h2)
    mlp_hidden = swiglu(gate, up)
    mlp_out = matvec(down_rows, mlp_hidden)
    return add(x, mlp_out)
