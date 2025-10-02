import torch
from torch import nn
from torch.nn import functional as F

def init_normal(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, mean=0, std=0.01)
		nn.init.zeros_(m.bias)

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net.apply(init_normal)
