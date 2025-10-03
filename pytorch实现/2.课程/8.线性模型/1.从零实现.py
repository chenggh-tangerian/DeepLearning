from d2l import torch as d2l
import random
import torch

# 1. 样本生成
def synthetic_data(w, b, num_examples):
	"""生成特定线性模型具有正态噪声的样本 x y"""
	num_features = len(w)

	w = torch.tensor(w)
	w.reshape(2, 1)
	x = torch.normal(0, 1, (num_examples, num_features))
	y = x @ w + b
	y += torch.normal(0, 0.01, y.shape)
	return x, y

# 2.数据存取
def data_iter(batch_size, features, labels):
	num_examples = len(features)
	idx = list(range(num_examples))
	# 打乱样本顺序
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		idx_batch = torch.tensor(idx[i: min(i + batch_size, num_examples)])
		yield features[idx_batch], labels[idx_batch]

# 3.初始化参数
class myModule:
	def __init__(self, w, b):
		self.w = torch.rand((len(w), 1), requires_grad=True)
		self.b = torch.zeros(1, requires_grad=True)
	# 4. 定义模型
	def linreg(self, X):
		"""线性回归模型"""
		return X @ self.w + self.b
	# 5. 定义损失函数
	@staticmethod
	def squared_loss(y_hat, y):
		# 导致梯度计算不正确
		y = y.reshape(y_hat.shape)
		return ((y_hat - y) ** 2) / 2
	# 6. 优化算法
	def sgd(self, lr, batch_size):
		with torch.no_grad():
			for param in (self.w, self.b):
				param -= lr * param.grad / batch_size
				param.grad.zero_()
	def forward(self, X):
		return self.linreg(X)

def main():
	w = [2, -3.4]
	b = 4.2
	batch_size = 10
	lr = 0.03
	epochs = 3
	net = myModule(w, b)
	loss = myModule.squared_loss

	features, labels = synthetic_data(w, b, 1000)
	for epoch in range(epochs):
		for X, y in data_iter(batch_size, features, labels):
			l = loss(net.forward(X), y)
			l.sum().backward()
			net.sgd(lr, batch_size)
		with torch.no_grad():
			train_l = loss(net.forward(features), labels)
			print(f'epoch {epoch + 1}, loss {train_l.mean():.5f}')
	print(net.w, net.b)

if __name__ == '__main__':
	main()