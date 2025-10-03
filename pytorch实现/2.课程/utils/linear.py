import torch

class Linear:
	def __init__(self):
		pass
	
	# 样本生成
	@staticmethod
	def synthetic_data(w, b, num_examples):
		"""生成特定线性模型具有正态噪声的样本 x y"""
		num_features = len(w)
		w = w.reshape(-1, 1)
		w = torch.tensor(w)
		x = torch.normal(0, 1, (num_examples, num_features))
		y = x @ w + b
		y += torch.normal(0, 0.01, y.shape)
		return x, y