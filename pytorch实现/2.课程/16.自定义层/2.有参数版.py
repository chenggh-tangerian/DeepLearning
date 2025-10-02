import torch
from torch import nn
from torch.nn import functional as F

class MyLayer(nn.Module):
	def __init__(self, in_units, units):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(in_units, units))
		self.bias = nn.Parameter(torch.randn(units))
	
	def forward(self, X):
		linear = torch.matmul(X, self.weight.data) + self.bias.data
		return F.relu(linear)
	
layer = MyLayer(5, 3)
Y = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print(Y)