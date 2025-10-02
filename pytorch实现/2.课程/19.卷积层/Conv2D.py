import torch
from torch import nn
from torch.nn import functional as F
from corr2d import corr2d

class Conv2D(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(kernel_size))
		self.bias = nn.Parameter(torch.zeros(1))
	def forward(self, x):
		return corr2d(x, self.weight) + self.bias
	
# 边缘检测（站在此处往右看）
# 0变成1是黑变白，卷积之后是-1
# 1变成0是白变黑，卷积之后是1
# 如果0的话，那说明这里不是边缘
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
# 但是只能看到右侧的变化，不能检测横向
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)