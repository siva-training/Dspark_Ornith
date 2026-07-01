"""A deliberately naive, fully-resident fp32 Ornith runner.

This exists only as a comparison point for ``dspark benchmark``: it loads
every tensor from the uncompressed source checkpoint into plain Python
lists up front and keeps the entire model resident for the whole run, the
way a straightforward "read the checkpoint, build the model" implementation
would. It is not part of dspark's public model API.
"""

from dspark.nn import DECODER_LAYER_WEIGHT_KEYS, argmax, new_kv_cache, rmsnorm, transformer_decoder_layer

from .config import OrnithConfig
from .source import OrnithSourceCheckpoint


class NaiveOrnithModel:
    def __init__(self, config, tensors):
        self.config = config
        self.tensors = tensors  # name -> (shape, flat floats), ALL resident at once
        self.kv_cache = new_kv_cache(config.num_layers)

    @classmethod
    def from_source(cls, path):
        with OrnithSourceCheckpoint(path) as src:
            config = OrnithConfig.from_dict(src.config)
            tensors = {name: src.read_tensor(name) for name in src.tensor_names()}
        return cls(config, tensors)

    def step(self, token_id, position):
        cfg = self.config
        h = cfg.hidden_size
        _, embed_flat = self.tensors["embed_tokens.weight"]
        x = embed_flat[token_id * h : (token_id + 1) * h]
        for layer_idx in range(cfg.num_layers):
            prefix = f"layers.{layer_idx}."
            weights = {key: self.tensors[prefix + key][1] for key in DECODER_LAYER_WEIGHT_KEYS}
            x = transformer_decoder_layer(x, weights, self.kv_cache[layer_idx], position, cfg)
        _, norm_w = self.tensors["norm.weight"]
        x = rmsnorm(x, norm_w, cfg.rms_norm_eps)
        lm_name = "lm_head.weight" if "lm_head.weight" in self.tensors else "embed_tokens.weight"
        _, lm_flat = self.tensors[lm_name]
        vocab = cfg.vocab_size
        return [sum(a * b for a, b in zip(lm_flat[i * h : (i + 1) * h], x)) for i in range(vocab)]

    def generate(self, prompt_ids, max_new_tokens):
        logits = None
        for pos, token_id in enumerate(prompt_ids):
            logits = self.step(token_id, pos)
        generated = []
        pos = len(prompt_ids)
        next_id = argmax(logits)
        for _ in range(max_new_tokens):
            generated.append(next_id)
            logits = self.step(next_id, pos)
            pos += 1
            next_id = argmax(logits)
        return generated
