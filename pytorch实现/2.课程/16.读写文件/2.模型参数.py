import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(20, 256)
		self.output = nn.Linear(256, 10)
	def forward(self, X):
		return self.output(F.relu(self.hidden(X)))
	
net = MLP()
X = torch.randn((2, 20))
Y = net(X)

# 定义文件,使用state_dict来存参数字典
torch.save(net.state_dict(), 'mlp.params')

# 读取文件,需要新建一个网络(不仅要存数据，还要带走定义方式)
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

# 验证完全一致
Y_clone = clone(X)
print(Y_clone == Y)