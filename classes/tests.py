import torch 

matrix = torch.rand((4, 2, 3))


print(matrix.shape)

print(matrix.transpose(-1, -2).shape)
