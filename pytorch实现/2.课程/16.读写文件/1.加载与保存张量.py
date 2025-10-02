import torch
from torch import nn
from torch.nn import functional as F

# 1. tensor
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)

# 2.dict
y = torch.zeros(4)
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)