import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """简单的全连接神经网络用于MNIST分类"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 展平输入 (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # 第一层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二层
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc3(x)
        return x