import torch
from torch import nn
from torch.nn import functional as F

class FixedHiddenMLP(nn.Module):
	def __init__(self):
		super().__init__()
		# 自定义行为
		self.rand_weight = torch.rand((20, 20), requires_grad=False)
		self.linear = nn.Linear(20, 20)
	def forward(self, X):
		X = self.linear(X)
		# 自定义矩阵加偏置
		X = F.relu(torch.mm(X, self.rand_weight) + 1)
		X = self.linear(X)
		return X
	
net = FixedHiddenMLP()
X = torch.rand((2, 20), requires_grad=True)
Y = net(X)
print(Y)