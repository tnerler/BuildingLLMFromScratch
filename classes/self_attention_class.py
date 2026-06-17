import torch.nn as nn 
import torch 

class SelfAttentionV1(nn.Module): 
    """
    Q: Query
    K: Key
    V: Value 

    1-) Starting the Q, K, V weights as random.

    2-) Calculating the matrix multiplication of all the weights (Q, K, V) with the input

    3-) Calculation the attention scores by applying (X @ Q) @ (X @ K).T

    4-) Calculation of the attention weights by applying softmax

    5-) Calculation of the context vector by (attention weights) @ (X @ V)
    """
    def __init__(self, d_in, d_out): 
        super().__init__()
        self.d_out = d_out 
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))
    
    def forward(self, x): 
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values 
        return context_vec
    

class SelfAttentionV2(nn.Module): 
    def __init__(self, d_in, d_out, qkv_bias=False): 
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
    
    def forward(self, x): 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attention_weights @ values 
        return context_vec 




if __name__ == "__main__": 
    import torch 
    torch.manual_seed(123)
    d_in = 3
    d_out = 2


    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your  (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
    )

    print(f"Self Attention 1 which is initalized weights with nn.Parameter(): \n")
    sa_v1 = SelfAttentionV1(d_in, d_out)
    print(sa_v1(inputs))
    print("*"*50)

    print(f"Self Attention 2 which is initalized weights with nn.Linear(): \n")
    sa_v2 = SelfAttentionV2(d_in, d_out)
    print(sa_v2(inputs))

    print("="*50)
    print(f"Assignin' the weights from an instance of SelfAttentionV2 to an instance of SelfAttentionV1.")


    print("Initalizing the weights from an instance SelfAttentionV2")

    weight_key_from_2 = sa_v2.W_key.weight.T # (2x3)
    weight_query_from_2 = sa_v2.W_query.weight.T  # (2x3)
    weight_value_from_2 = sa_v2.W_value.weight.T  # (2x3)

    print(f"Initalized the weight.")
    print(f"`Key Weight` from an instace of SelfAttentionV2:\n {weight_key_from_2}")
    print(f"`Query Weight` from an instace of SelfAttentionV2:\n {weight_query_from_2}")
    print(f"`Value Weight` from an instace of SelfAttentionV2:\n {weight_value_from_2}")


    print("Assigning these weights to an instance of SelfAttentionV1")

    sa_v1.W_key.data = weight_key_from_2
    sa_v1.W_query.data = weight_query_from_2
    sa_v1.W_value.data = weight_value_from_2

    print("Assigned the weights to SelfAttentionV1")
    print(f"Initalized the weight.")
    print(f"`Key Weight` from an instace of SelfAttentionV1: {sa_v1.W_key}")
    print(f"`Query Weight` from an instace of SelfAttentionV1: {sa_v1.W_query}")
    print(f"`Value Weight` from an instace of SelfAttentionV1: {sa_v1.W_value}")


    print("Implementing the same inputs\n\n")

    print("SelfAttentionV1 with inputs:\n")
    print(sa_v1(inputs))

    print("SelfAttentionV2 with inputs")
    print(sa_v2(inputs))

    print("Being sure if these are the same results\nSince we assign the weights from the V2 to V1")
    print(torch.allclose(sa_v1(inputs), sa_v2(inputs)))