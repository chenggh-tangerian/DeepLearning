import torch
from torch import nn
from torch.nn import functional as F

shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU()
					,nn.Linear(8, 1))
X = torch.rand(4, 4)

Y = net(X)
# print(net[2].weight.data == net[2].weight.data)
print((net[2].weight.data == net[2].weight.data).sum())

net[2].weight.data += 100
print((net[2].weight.data == net[2].weight.data).sum())