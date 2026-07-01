"""Base config/model contracts shared by every dspark model family."""

import abc
from dataclasses import asdict, dataclass


@dataclass
class BaseDsparkConfig:
    """Fields common to every decoder-only family dspark supports.

    Concrete families (e.g. Ornith) subclass this; sharing the base keeps
    conversion tooling and the CLI family-agnostic.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    intermediate_size: int
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads (GQA groups)")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class BaseDsparkModel(abc.ABC):
    """Contract every registered dspark model must satisfy."""

    family_name = None

    @classmethod
    @abc.abstractmethod
    def from_checkpoint(cls, checkpoint, max_resident_layers=None):
        """Build a runnable model from a :class:`dspark.storage.DsparkCheckpoint`."""

    @abc.abstractmethod
    def generate(self, prompt_ids, max_new_tokens):
        """Greedy-decode ``max_new_tokens`` token ids following ``prompt_ids``."""
