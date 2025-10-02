import torch
from torch import nn
from torch.nn import functional as F

def block1():
	return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
	net = nn.Sequential()
	for i in range(4):
		net.add_module(f'block {i}', block1())
	return net

X = torch.rand(2, 4)
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
Y = rgnet(X)

print(rgnet)