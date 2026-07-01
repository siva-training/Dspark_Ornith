from dspark.registry import register_model_family

from .config import OrnithConfig
from .convert import convert_source_to_dspark
from .model import OrnithForCausalLM

register_model_family(
    name="ornith",
    config_cls=OrnithConfig,
    model_cls=OrnithForCausalLM,
    converter=convert_source_to_dspark,
)

__all__ = ["OrnithConfig", "OrnithForCausalLM", "convert_source_to_dspark"]
