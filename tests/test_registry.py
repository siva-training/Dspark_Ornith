import unittest

import dspark  # noqa: F401  (import triggers registration of built-in families)
from dspark.models.ornith.config import OrnithConfig
from dspark.models.ornith.model import OrnithForCausalLM
from dspark.registry import get_model_family, registered_families


class TestRegistry(unittest.TestCase):
    def test_ornith_is_registered(self):
        self.assertIn("ornith", registered_families())

    def test_get_model_family_returns_wired_classes(self):
        family = get_model_family("ornith")
        self.assertIs(family.config_cls, OrnithConfig)
        self.assertIs(family.model_cls, OrnithForCausalLM)
        self.assertTrue(callable(family.converter))

    def test_unknown_family_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_model_family("not-a-real-family")


if __name__ == "__main__":
    unittest.main()
