import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

class MNISTNet(nn.Module):
    """简单的全连接神经网络用于MNIST分类"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10, T=6):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_s = tdLayer(self.fc1)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_s = tdLayer(self.fc2)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc3_s = tdLayer(self.fc3)
        
        # self.dropout = nn.Dropout(0.2)
        self.act = LIFSpike()
        self.T = T
        
    def forward(self, x):
        # 展平输入 (batch_size, 28, 28) -> (batch_size, 784)
        # 开始x [1, 1, 28, 28] # 为什么四维？
        x = x.view(x.size(0), -1) # ([1, 784])
        
        # 增加时间维度 [batch, 28 * 28]
        x = add_dimention(x, self.T) # 增加一个时间维度，大小为T
        # x [1, T6, 784]

        # 第一层
        x = self.act(self.fc1_s(x)) # 输出只有0和1
        # x = self.dropout(x)
        
        # 第二层
        x = self.act(self.fc2_s(x))
        # x = self.dropout(x)
        
        # 输出层
        x = self.fc3_s(x)
        x = x.mean(1) # 第一位取平均（降维）
        return x