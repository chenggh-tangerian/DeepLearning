import torch
import torch.nn 
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
from utils import FashionMnistManager

manager = FashionMnistManager()
batch_size = 256
num_inputs = manager.num_inputs
num_outputs = manager.num_outputs
lr = 0.03

train_iter, test_iter = manager.get_train_iter(batch_size), manager.get_test_iter(batch_size)
W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
	"""X: (b, f)"""
	X_exp = torch.exp(X)
	X_sum = X_exp.sum(1, keepdim=True)
	return X_exp / X_sum

# X = torch.normal(0, 1, (2, 5), requires_grad=False)
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

def net(X):
	Y = torch.matmul(X.reshape((-1, W.shape[0])), W) + b
	return softmax(Y)

def cross_entropy(y_hat, y):
	return -torch.log(y_hat[range(len(y_hat)), y])

def acc(y_hat, y):
	y_hat = y_hat.argmax(1) # 最大值的标号
	cmp = (y_hat == y)
	return float(cmp.sum() / len(y))

def test():
	features = torch.tensor([[1.65, 1.10, 0.52], [5.3, 6.5, 9.0]])
	y = torch.tensor([0, 2])
	y_hat = softmax(features)
	print(y_hat)
	print(acc(y_hat, y))

loss = cross_entropy
def updater(batch_size):
	return d2l.sgd([W, b], lr, batch_size)
updater = updater

epochs = 3
for epoch in range(epochs):
	for X, y in train_iter:
		y_hat = net(X)
		l = loss(y_hat, y)
		l.sum().backward()
		updater(X.shape[0])
	acc_test, block = 0, 0
	for i, (X, y) in enumerate(test_iter):
		block = i + 1
		y_hat = net(X)
		acc_test += acc(y_hat, y)
	acc_test /= (block)
	print(acc_test)



