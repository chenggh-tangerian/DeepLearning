import torch
from torch import nn
from torch.nn import functional as F

# stride = 1, padding = 0
# 算子
def corr2d(X, K):
	"""计算二维互相关运算"""
	h, w = K.shape
	Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	# 按照公式实现i, j是左上角元素
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			# 按元素相乘，之后使用sum()进行相加返回
			Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
	return Y

X = torch.arange(9).reshape(3, 3)
K = torch.arange(4).reshape(2, 2)
Y = corr2d(X, K)
print(Y)

