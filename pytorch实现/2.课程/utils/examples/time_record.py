# 验证向量化加速 ???
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))  # .../utils
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from time_recorder import TimeRecorder

N = 1000

def main():
	a = torch.FloatTensor(N)
	b = torch.FloatTensor(N)
	c = torch.zeros_like(a)
	d = torch.zeros_like(a)

	timer = TimeRecorder()

	for i in range(N):
		c[i] = a[i] + b[i]
	timer.stop()

	d = a + b
	timer.stop()

	for idx, time in enumerate(timer.times):
		print(f"timer.times{idx}: {time:.3f}")

	# 用法示例
	for idx, time in enumerate(timer.cumsum()):
		print(f"timer.times{idx}: {time:.3f}")
	print(f"timer.sum: {timer.sum():.3f}")
	print(f"timer.avg: {timer.avg():.3f}")

if __name__ == '__main__':
	main()