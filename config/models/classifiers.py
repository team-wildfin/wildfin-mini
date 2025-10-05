from config.models.models import ModelConfig
from typing import Dict
from torch import nn


LinearHead = ModelConfig(
    name="linear",
    input_ndim=1,
    output_ndim=1,
) 

MLPHead = ModelConfig(
    name="mlp",
    input_ndim=1,
    output_ndim=1,
)

CLASSIFIER_CONFIGS: Dict[str, ModelConfig] = {
    "linear": LinearHead,
    "mlp": MLPHead,
} 

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers):
        super().__init__()
        assert num_layers >= 2, "MLP must have at least 2 layers"
        self.mlp = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim), 
            nn.ReLU(),
            *[layer for _ in range(num_layers - 2) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
            nn.Linear(hidden_dim, out_features)
        ])
    
    def forward(self, x):
        return self.mlp(x)

CLASSIFIER_MODULES: Dict[str, nn.Module] = {
    "linear": nn.Linear,
    "mlp": MLP,
}