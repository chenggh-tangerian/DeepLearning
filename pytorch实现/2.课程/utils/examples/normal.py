import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../utils
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from functions import Functions as F
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (0, 3)]
y = [F.normal(x, mu, sigma) for mu, sigma in params]
y = np.array(y)

print("x 的形状:", x.shape)
print("y 值的数量:", y.shape)
print(y)

d2l.plot(x, y, xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
		 legend=[f'main {mu}, std {sigma}' for mu, sigma in params])
plt.show()
print("OK")

import time 
time.sleep(30)