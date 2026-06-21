from torch import nn 
import torch
torch.manual_seed(123)

torch.manual_seed(123)

input_1 = torch.randn(size=(1, 2, 3, 4))
input_2 = torch.randn(size=(1, 2, 3, 4))



transposed_1 = input_1.transpose(1, 2)
transposed_2 = input_2.transpose(1, 2)

result_1 = transposed_1 @ transposed_2.transpose(2, 3)

for batch in input_1: 
    for idx, token in enumerate(batch): 
        for idx_, head in enumerate(token): 
            print(f"Head {idx_} for Token {idx}: {head}")

