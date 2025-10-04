import torch
import torch.nn 
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 读取数据集
# 数据集处理器
trans = transforms.ToTensor()
# 数据集
mnist_train = torchvision.datasets.FashionMNIST(
	root="D:\Projects\github\DeepLearning\pytorch实现\data", train=True, download=False, transform=trans
)
mnist_test = torchvision.datasets.FashionMNIST(
	root="D:\Projects\github\DeepLearning\pytorch实现\data", train=False, download=False, transform=trans
)

# 根据分类标签 获得文本标签
def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
				'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]

# 使用plt画出图片
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
	# figsize = (num_rows * scale, num_cols * scale)
	_, axes = d2l.plt.subplots(num_rows, num_cols)
	axes = axes.flatten()
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		if torch.is_tensor(img):
			ax.imshow(img.numpy())
		else:
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
	plt.show()
	return axes

# 加载器的使用
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 批量读取
batch_size = 256
def get_dataloader_workers():
	return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
							 num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
	continue

print(f'{timer.stop():.2f} sec')
