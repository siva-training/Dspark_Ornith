"""Pluggable model-family registry.

dspark is not written for one model - it is a small runtime that any
transformer family sharing its core primitives (see :mod:`dspark.nn`) can
plug into by registering a config class, a model class, and a checkpoint
converter under a family name. Ornith registers itself here on import
(see ``dspark.models.ornith``); a sibling model from the same architectural
family could be added the same way without touching dspark's core.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ModelFamily:
    name: str
    config_cls: type
    model_cls: type
    converter: Callable[..., Any]


_FAMILIES: Dict[str, ModelFamily] = {}


def register_model_family(name, config_cls, model_cls, converter):
    if name in _FAMILIES:
        raise ValueError(f"model family {name!r} is already registered")
    _FAMILIES[name] = ModelFamily(name, config_cls, model_cls, converter)


def get_model_family(name):
    try:
        return _FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(_FAMILIES)) or "<none registered>"
        raise KeyError(f"unknown dspark model family {name!r}. Registered families: {known}") from None


def registered_families():
    return sorted(_FAMILIES)


def _reset_for_tests():
    """Test-only helper to clear the registry between isolated unit tests."""
    _FAMILIES.clear()
