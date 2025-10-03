from d2l import torch as d2l
import random
import torch
from torch.utils import data
import torch.nn as nn

import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../utils
ROOT = os.path.join(ROOT, 'utils')
sys.path.append(ROOT)

from utils import Linear

def load_array(data_arrays, batch_size, is_train=True):
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

def main():
	w = torch.tensor([2, -3.4])
	b = 4.2
	batch_size = 10
	epochs = 3

	# 数据需要被reshape成列向量！！！
	features, labels = Linear.synthetic_data(w, b, 1000)
	data_iter = load_array((features, labels), batch_size)
	net = nn.Sequential(nn.Linear(2, 1))
	net[0].weight.data.normal_(0, 0.01)
	net[0].bias.data.fill_(0)

	loss = nn.MSELoss()
	trainer = torch.optim.SGD(params=net.parameters(), lr=0.03)

	for epoch in range(epochs):
		for X, y in data_iter:
			y_hat = net(X)
			y = y.reshape(y_hat.shape)
			l = loss(y_hat, y)
			trainer.zero_grad()
			l.backward()
			trainer.step()
		with torch.no_grad():
			train_l = loss(net(features), labels)
			print(f'epoch {epoch + 1}, loss {train_l.mean():.5f}')
	print(net[0].weight.data, net[0].bias.data)
	

if __name__ == '__main__':
	main()