import numpy as np
import torch

# a = torch.randn(8, 4)
# b = torch.ones(8, 1).long()

# c = a.gather(1, b)

# print(a)
# print(b)
# print(c)

# a = [[0.1, 0.2, 0.3, 0.4] for _ in range(8)]
# a = np.asarray(a)
# a = torch.from_numpy(a)

# b = torch.ones(8, 1).long()
# c = a.gather(1, b)

# print(a)
# print(b)
# print(c)

# cont_feat = torch.ones(8, 32, 32, 32)
# sty_feat = torch.ones(1, 32, 32, 32)
# weight = torch.ones(8, 1)

# weight = weight.unsqueeze(2).unsqueeze(3)
# targetFeature = (1.0 - weight) * cont_feat + weight * sty_feat

# print(targetFeature)