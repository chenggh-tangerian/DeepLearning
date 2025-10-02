import torch
from torch import nn
from torch.nn import functional as F
# 任何一个层都是module子类
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
Y = net(X)

print(Y)

