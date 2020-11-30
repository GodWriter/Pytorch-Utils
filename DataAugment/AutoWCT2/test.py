import torch

a = torch.randn(2, 2, 2)
b = torch.stack((a, a), 2)

print(a)
print(b)
print(torch.squeeze(b, 3))