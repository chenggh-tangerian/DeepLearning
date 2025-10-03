import torch
import time
import numpy as np

class TimeRecorder:
	def __init__(self):
		self.times = []
		self.start()
	
	def start(self):
		"""启动计时器"""
		self.tik = time.time()
	
	def stop(self):
		"""停止计时器并将时间保存到time中"""
		self.times.append(time.time() - self.tik)
		return self.times[-1]
	
	def avg(self):
		"""返回平均时间"""
		return sum(self.times) / len(self.times)
	
	def sum(self):
		"""返回时间总和"""
		return sum(self.times)
	
	def cumsum(self):
		"""返回累计时间"""
		return np.array(self.times).cumsum().tolist()