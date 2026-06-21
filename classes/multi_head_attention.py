import torch 
from torch import nn 

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, context_length, num_heads, dropout, bias=False): 
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisable by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model 
        self.head_dim = self.d_model // self.num_heads

        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=bool), diagonal=1)
        )

    
    def forward(self, X): 
        batch_size, seq_length, dim = X.shape

        keys = self.W_k(X)
        queries = self.W_q(X)
        values = self.W_v(X)

        
        # (B, L, D) -> (B, L, H, Dh)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # (B, L, H, Dh) -> (B, H, L, Dh)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        # apply mask 

        mask = self.mask[:seq_length, :seq_length]
        attn_scores.masked_fill_(mask, -torch.inf)

        # compute attention weights
        attn_weights = torch.softmax(
            attn_scores / self.head_dim ** 0.5, dim=-1
        )
        
        # apply dropout to reduce overfitting 
        attn_weights = self.dropout(attn_weights)

        # compute the context vec
        context_vecs = (attn_weights @ values).transpose(1, 2)
        context_vecs = context_vecs.contiguous().view(batch_size, seq_length, self.d_model)
        context_vecs = self.W_o(context_vecs)
        return context_vecs




if __name__ == "__main__": 
    context_length = 1024
    attention_head = 12
    d_model = 768


    inputs = torch.randn(size=(2, 100, 768))

    mh = MultiHeadAttention(d_model, context_length, attention_head, 0.0)

    print(f"Input Shape:\n{inputs.shape}\nInputs: {inputs}")
    print()
    print(f"Multi Head Attention Context Vector Shape:\n")
    context_vecs = mh(inputs)
    print(context_vecs.shape)
    print()
    print(f"Context Vector: {context_vecs}")