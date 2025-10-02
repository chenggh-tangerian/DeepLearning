import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, X):
		return X - X.mean()
	
layer = CenteredLayer()
Y = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print(Y)