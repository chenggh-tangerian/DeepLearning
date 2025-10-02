import torch
from torch import nn
from torch.nn import functional as F

# Sequential是python的一个list
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand((2, 4), requires_grad=True)
Y = net(X)
print(Y)

# 现在从list里拿东西
print(net[2].state_dict())

print(net[2].bias)
# <class 'torch.nn.parameter.Parameter'> 可优化的参数
print(type(net[2].bias))
print(net[2].bias.data)
print(net[2].bias.grad == None)

# 一次拿出所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['0.weight'].data)