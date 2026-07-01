"""OrnithForCausalLM: a decoder-only transformer that runs directly off a
memory-mapped, quantized .dsq checkpoint with a bounded memory footprint.

Three things keep peak memory low regardless of total model size:

1. The checkpoint is memory-mapped (see :mod:`dspark.storage`) - weight
   bytes live in OS page cache, and disk-backed pages the OS hasn't
   touched yet cost no resident memory at all.
2. The token embedding and (if untied) lm_head are looked up one row at a
   time via ``LazyTensor.get_row`` - cost is independent of vocab size.
3. Decoder layers are decoded to Python floats lazily and cached in an
   LRU of size ``max_resident_layers``; once that many layers are
   resident, the oldest is evicted before a new one is decoded. This is
   the direct analogue of llama.cpp-style layer offloading, letting a
   model with far more layers than fit in RAM still run to completion.
"""

from collections import OrderedDict

from dspark.base import BaseDsparkModel
from dspark.nn import DECODER_LAYER_WEIGHT_KEYS, argmax, new_kv_cache, rmsnorm, transformer_decoder_layer

from .config import OrnithConfig

DEFAULT_MAX_RESIDENT_LAYERS = 2


class OrnithForCausalLM(BaseDsparkModel):
    family_name = "ornith"

    def __init__(self, config, checkpoint, max_resident_layers=DEFAULT_MAX_RESIDENT_LAYERS):
        if max_resident_layers < 1:
            raise ValueError("max_resident_layers must be >= 1")
        self.config = config
        self.checkpoint = checkpoint
        self.max_resident_layers = max_resident_layers
        self._layer_cache = OrderedDict()  # layer_idx -> {weight_name: flat floats}
        self._embed_tensor = checkpoint.get_tensor("embed_tokens.weight")
        self._norm = checkpoint.get_tensor("norm.weight").load()
        lm_head_name = "lm_head.weight" if "lm_head.weight" in checkpoint.tensor_names() else None
        self._lm_head_tensor = checkpoint.get_tensor(lm_head_name) if lm_head_name else self._embed_tensor
        self.kv_cache = new_kv_cache(config.num_layers)

    @classmethod
    def from_checkpoint(cls, checkpoint, max_resident_layers=DEFAULT_MAX_RESIDENT_LAYERS):
        if checkpoint.model_family != "ornith":
            raise ValueError(
                f"checkpoint is for model family {checkpoint.model_family!r}, not 'ornith'"
            )
        config = OrnithConfig.from_dict(checkpoint.config)
        return cls(config, checkpoint, max_resident_layers=max_resident_layers)

    def resident_layer_indices(self):
        return list(self._layer_cache.keys())

    def _get_layer_weights(self, layer_idx):
        if layer_idx in self._layer_cache:
            self._layer_cache.move_to_end(layer_idx)
            return self._layer_cache[layer_idx]
        prefix = f"layers.{layer_idx}."
        weights = {key: self.checkpoint.get_tensor(prefix + key).load() for key in DECODER_LAYER_WEIGHT_KEYS}
        self._layer_cache[layer_idx] = weights
        if len(self._layer_cache) > self.max_resident_layers:
            self._layer_cache.popitem(last=False)  # evict least-recently-used layer
        return weights

    def _lm_head_logits(self, x):
        vocab = self.config.vocab_size
        return [sum(a * b for a, b in zip(self._lm_head_tensor.get_row(i), x)) for i in range(vocab)]

    def step(self, token_id, position):
        """Run one autoregressive step, mutating ``self.kv_cache``."""
        cfg = self.config
        x = self._embed_tensor.get_row(token_id)
        for layer_idx in range(cfg.num_layers):
            weights = self._get_layer_weights(layer_idx)
            x = transformer_decoder_layer(x, weights, self.kv_cache[layer_idx], position, cfg)
        x = rmsnorm(x, self._norm, cfg.rms_norm_eps)
        return self._lm_head_logits(x)

    def generate(self, prompt_ids, max_new_tokens):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
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
