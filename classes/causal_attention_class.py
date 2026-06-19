import torch 
from torch.nn import Linear, Module 
from torch import nn 

class CausalAttention(Module): 
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__() 
        self.d_out = d_out 

        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1) # register_buffer which we use to store the masks, it includes in the model's state_dict but not include 
                        # as the model parameter to train it. So, we do not want to train the mask tensor but we want to store it
                        # to use, so we store it like this.
        )

    def forward(self, x): 
        _, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-1, -2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values 
        return context_vec
    

if __name__ == "__main__": 
    torch.manual_seed(123)
    
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your  (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
    )


    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]

    d_in = inputs.shape[1] 
    d_out = 2
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch) 

    print(f"context_vec:\n{context_vecs}")
    print(f"context_vecs.shape: {context_vecs.shape}")
