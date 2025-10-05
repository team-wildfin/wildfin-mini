from config.models.models import ModelConfig
from torch import nn
import torch
from typing import Dict

ATTN_POOLER = ModelConfig(
    name="attention",
    category="pooling",
    input_ndim=2,
    output_ndim=1,
)
MEAN_POOLER = ModelConfig(
    name="mean",
    category="pooling",
    input_ndim=2,
    output_ndim=1,
)
MAX_POOLER = ModelConfig(
    name="max",
    category="pooling",
    input_ndim=2,
    output_ndim=1,
) 

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.query_token.expand(B, -1, -1)  # [B, 1, D]
        x = self.norm(x)
        attn_out, _ = self.attn(q, x, x)
        return attn_out.squeeze(1)  # [B, D]

class MeanPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.mean(dim=self.dim)
    
class MaxPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim).values


POOLER_MODULES: Dict[str, nn.Module] = {
    "attention": AttentionPooling,
    "mean": MeanPooling,
    "max": MaxPooling,
}