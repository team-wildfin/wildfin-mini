import unittest
import torch
from vision_bench.models import (
    BackBoneModel,
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    MLP,
    Linear,
    BroadcastableModule,
    ComposedModel,
    ModelBuilder
)

class TestModelComponents(unittest.TestCase):

    def test_backbone_output_shape(self):
        model = BackBoneModel("dino")
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.ndim, 3)  # [batch, tokens, embed_dim]

    def test_mean_pooling(self):
        pool = MeanPooling(dim=1)
        x = torch.randn(8, 16, 384)
        out = pool(x)
        self.assertEqual(out.shape, (8, 384))

    def test_max_pooling(self):
        pool = MaxPooling(dim=1)
        x = torch.randn(8, 16, 384)
        out = pool(x)
        self.assertEqual(out.shape, (8, 384))

    def test_attention_pooling(self):
        pool = AttentionPooling(embed_dim=384)
        x = torch.randn(8, 16, 384)
        out = pool(x)
        self.assertEqual(out.shape, (8, 384))

    def test_mlp_forward(self):
        mlp = MLP(input_dim=384, hidden_dim=512, output_dim=10, num_layers=2)
        x = torch.randn(8, 384)
        out = mlp(x)
        self.assertEqual(out.shape, (8, 10))

    def test_linear_forward(self):
        linear = Linear(input_dim=384, output_dim=10)
        x = torch.randn(8, 384)
        out = linear(x)
        self.assertEqual(out.shape, (8, 10))

    def test_broadcastable_module(self):
        base = Linear(384, 10)
        base.input_ndim = 2  # simulate interface
        wrapped = BroadcastableModule(base)
        x = torch.randn(2, 3, 384)
        out = wrapped(x)
        self.assertEqual(out.shape, (2, 3, 10))

    def test_composed_model_end_to_end(self):
        backbone = BroadcastableModule(BackBoneModel("dino"))
        embed_dim = backbone(torch.randn(1, 3, 224, 224)).shape[-1]
        pooling = BroadcastableModule(MeanPooling(dim=1))
        classifier = BroadcastableModule(Linear(embed_dim, 5))
        model = ComposedModel(backbone, pooling, classifier)

        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_model_builder(self):
        builder = ModelBuilder()
        hidden_size = builder.set_model("dino").get_hidden_size()
        builder.set_model("dino").set_pooling("mean").set_classifier("linear", input_dim=hidden_size, output_dim=5)
        model = builder.build()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))


if __name__ == '__main__':
    unittest.main()
