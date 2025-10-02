import torch
from torch import nn
from torch.nn import functional as F

class MySequential(nn.Module):
	# 将需要给出的在构造时给出，用于自定义
	def __init__(self, *args):
		super().__init__()
		for block in args:
			self._modules[block] = block

	def forward(self, X):
		for block in self._modules.values():
			X = block(X)
		return X
	
X = torch.rand(2, 20)
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
Y = net(X)
print(Y)