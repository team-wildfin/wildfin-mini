from config.models.models import ModelConfig
from torch import nn
import torch
from typing import Dict
from config.models.vjepa_module import CrossAttention, Block, CrossAttentionBlock
from config.models.vjepa_utils import trunc_normal_
import math 


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
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        if hidden_size % num_heads != 0:
            num_heads = 1  # Fallback to single head if hidden_size not divisible
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.query_token.expand(B, -1, -1)  # [B, 1, D]
        x = self.norm(x)
        attn_out, _ = self.attn(q, x, x)
        return attn_out.squeeze(1)  # [B, D]

class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        hidden_size=1024,
        num_heads=16,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_size))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if q.shape[1] == 1:
            q = q.squeeze(1)
        return q
    

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
    "vjepa2_attention": AttentivePooler,
    "mean": MeanPooling,
    "max": MaxPooling,
}