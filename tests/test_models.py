from __future__ import annotations

import unittest
import warnings

import torch

from models import MODEL_NAMES, build_model


class ModelSmokeTests(unittest.TestCase):
    def test_all_models_forward(self) -> None:
        inputs = torch.zeros(2, 14, 128)
        for name in MODEL_NAMES:
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_model(name, 14, 128, 4, 25.0)
                model.eval()
                with torch.no_grad():
                    output = model(inputs)
                self.assertEqual(tuple(output.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
