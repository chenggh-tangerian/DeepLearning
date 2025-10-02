import torch
from torch import nn
from torch.nn import functional as F

from MyModule import MySequential
class NestMLP(nn.Module):
	def __init__(self, *args):
		super().__init__()
		self.net = MySequential(*args)
		self.linear = nn.Linear(10, 10)
	def forward(self, X):
		return self.linear(self.net(X))
	
X = torch.rand((2, 20), requires_grad=True)
net = NestMLP(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10), nn.ReLU())
Y = net(X)