import torch

A = torch.ones(size=torch.Size([3, 8, 16]))  # i,k
B = torch.ones(size=torch.Size([3, 9, 16]))  # j,k

print(torch.matmul(A, B.permute(0, 2, 1)).shape)