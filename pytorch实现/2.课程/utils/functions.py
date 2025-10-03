import torch
import math
import numpy as np

import d2l

class Functions:
	def __init__():
		pass

	@staticmethod
	def normal(x, mu, sigma):
		p = 1 / math.sqrt(2 * math.pi * sigma**2)
		return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)