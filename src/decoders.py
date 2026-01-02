import torch.nn as nn
from .abstract import BaseDecoder

class StandardMLP(BaseDecoder):
    """标准 MLP，用于 Part 1 和 Part 2"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, num_layers=3):
        super().__init__()
        
        # 动态构建网络层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 添加隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Part 1 输出 RGB 需要 Sigmoid
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)