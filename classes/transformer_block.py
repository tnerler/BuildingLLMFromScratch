import torch 
from torch import nn 
from classes.multi_head_attention import MultiHeadAttention
from classes.feed_forward import FeedForward
from classes.layer_norm import LayerNorm
import json 


class TransformerBlock(nn.Module): 
    def __init__(self, cfg): 
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length= cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x): 
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
if __name__ == "__main__": 
    torch.manual_seed(123)
    x = torch.randn(2, 4, 768)


    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    block = TransformerBlock(cfg)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    