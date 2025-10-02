import torch
from torch import nn
from torch.nn import functional as F
from Conv2D import Conv2D
from corr2d import corr2d

X = torch.ones(6, 8)
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K).reshape(1, 1, 6, 7)
X = X.reshape(1, 1, 6, 8)
# 通道数（黑白为1，彩色为3）
# batch数，这里我们一个一个训练
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

for i in range(10):
	Y_hat = conv2d(X)
	l = (Y_hat - Y)**2
	conv2d.zero_grad()
	l.sum().backward()
	conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
	if (i + 1) % 2 == 0:
		print(f'batch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape(1, 2))