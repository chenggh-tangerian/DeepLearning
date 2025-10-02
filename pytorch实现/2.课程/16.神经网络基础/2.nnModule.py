import torch
from torch import nn
from torch.nn import functional as F

# 多层感知机
# 自定义nn.Module类
class MLP(nn.Module):
	# 需要哪些类哪些参数
	def __init__(self):
		# 父类
		super().__init__()
		# 全连接层，隐藏层
		self.hidden = nn.Linear(20, 256)
		# 输出层
		self.out = nn.Linear(256, 10)
	
	def forward(self, X):
		return self.out(F.relu(self.hidden(X)))
	
def main():
	X = torch.rand(3, 20)
	# 实例化类
	net = MLP()
	# 放入输入
	Y = net(X)

	print(Y)

if __name__ == '__main__':
	main()