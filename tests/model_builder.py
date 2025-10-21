from fish_benchmark.models import ModelBuilder
import unittest
from typing import Dict
import torch
from config.maps.backbone_preprocessors import PREPROCESSORS

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        # Each dict corresponds to a call to ModelBuilder.build(...)
        self.test_cases = [
            # Backbone only (CNN)
            {"backbone_name": "dinov3_large", "pooler_name": None, "classifier_name": None,
             "aggregator_name": None, "hidden_size": None, "output_dim": None, "freeze_backbone": True},

            # Backbone + Pooler (CNN + MeanPool)
            {"backbone_name": "dinov3_large", "pooler_name": "mean", "classifier_name": None,
             "aggregator_name": None, "hidden_size": None, "output_dim": None, "freeze_backbone": False},

            # Backbone + Pooler (Transformer + AttentionPool)
            {"backbone_name": "videomae", "pooler_name": "attention", "classifier_name": None,
             "aggregator_name": None, "hidden_size": 768, "output_dim": None, "freeze_backbone": True},

            # Classifier only (Linear)
            {"backbone_name": None, "pooler_name": None, "classifier_name": "linear",
             "aggregator_name": None, "hidden_size": 128, "output_dim": 10, "freeze_backbone": None},

            # Classifier + Aggregator (mlp + MeanPool)
            {"backbone_name": None, "pooler_name": None, "classifier_name": "mlp",
             "aggregator_name": "mean", "hidden_size": 128, "output_dim": 5, "freeze_backbone": None},

            # Full build: Transformer + Attention + mlp + Mean
            {"backbone_name": "videomae", "pooler_name": "attention", "classifier_name": "mlp",
             "aggregator_name": None, "hidden_size": 768, "output_dim": 10, "freeze_backbone": False},

            # Full build: CNN + Mean + Linear + Identity aggregator
            {"backbone_name": "dinov3_large", "pooler_name": "mean", "classifier_name": "linear",
             "aggregator_name": None, "hidden_size": 1024, "output_dim": 3, "freeze_backbone": True},

            {'backbone_name': 'vjepa2', 'pooler_name': 'attention', 'classifier_name': 'mlp',
             'aggregator_name': None, 'hidden_size': None, 'output_dim': 10, 'freeze_backbone': True},
        ]

    def test_build(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                try:
                    model = ModelBuilder.build(**case)
                    self.assertIsNotNone(model)  # just check it ran
                except Exception as e:
                    self.fail(f"ModelBuilder.build raised an exception {e} for case {case}")
    
    def test_data_pass(self): 
        pass

if __name__ == "__main__":
    unittest.main()