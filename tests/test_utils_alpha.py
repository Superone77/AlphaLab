import importlib
import os
import sys
import types
import unittest

import torch


class AlphaHillFixFingerTest(unittest.TestCase):
    def setUp(self):
        self._old_fix_finger = os.environ.get("FIX_FINGER")
        os.environ.pop("FIX_FINGER", None)

    def tearDown(self):
        if self._old_fix_finger is None:
            os.environ.pop("FIX_FINGER", None)
        else:
            os.environ["FIX_FINGER"] = self._old_fix_finger

    def _reload_utils(self):
        if "loguru" not in sys.modules:
            loguru = types.ModuleType("loguru")
            loguru.logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                                 warning=lambda *a, **k: None)
            sys.modules["loguru"] = loguru

        import alphalab.utils_alpha as utils_alpha

        return importlib.reload(utils_alpha)

    def test_fix_finger_defaults_to_xmin_peak(self):
        utils_alpha = self._reload_utils()

        self.assertEqual(utils_alpha.FIX_FINGER, "xmin_peak")

    def test_fix_finger_can_be_disabled_from_environment(self):
        os.environ["FIX_FINGER"] = "none"
        utils_alpha = self._reload_utils()

        self.assertIsNone(utils_alpha.FIX_FINGER)

    def test_alpha_hill_uses_fix_finger_by_default(self):
        utils_alpha = self._reload_utils()
        eigs = torch.tensor(
            [0.2, 0.25, 0.28, 0.3, 0.33, 3.0, 3.5, 4.1, 5.0, 6.0],
            dtype=torch.float32,
        )

        default_alpha, default_k, _ = utils_alpha._alpha_from_sorted_eigs(eigs)
        fixed_alpha, fixed_k, _ = utils_alpha._alpha_from_sorted_eigs(
            eigs, fix_finger="xmin_peak"
        )
        top_k_alpha, top_k, _ = utils_alpha._alpha_from_sorted_eigs(
            eigs, fix_finger=None, k=2
        )

        self.assertEqual((default_alpha, default_k), (fixed_alpha, fixed_k))
        self.assertNotEqual((default_alpha, default_k), (top_k_alpha, top_k))


if __name__ == "__main__":
    unittest.main()
