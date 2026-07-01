"""dspark: a dependency-free, memory-mapped, quantized inference runtime.

dspark exists to run models from a shared decoder-only transformer family
(RMSNorm + RoPE + grouped-query attention + SwiGLU - the family Ornith
belongs to) within a bounded, small memory budget suitable for consumer
hardware, via three complementary techniques:

1. Blockwise int8/int4 quantization of weights (see :mod:`dspark.quant`).
2. A memory-mapped checkpoint container so weights live in OS page cache,
   not process heap, until actually read (see :mod:`dspark.storage`).
3. Bounded layer streaming: only a small, configurable number of decoder
   layers are ever decoded into Python objects at once (see
   ``dspark.models.ornith.model``).

Importing this package registers every built-in model family (currently
just Ornith) with :mod:`dspark.registry`.
"""

from . import models  # noqa: F401  (registers built-in model families)
from .registry import get_model_family, registered_families

__all__ = ["get_model_family", "registered_families"]

__version__ = "0.1.0"
