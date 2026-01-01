import torch.nn as nn
from .abstract import BaseDecoder

class StandardMLP(BaseDecoder):
    """标准 MLP，用于 Part 1 和 Part 2"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # Part 1 输出 RGB 需要 Sigmoid
        )

    def forward(self, x):
        return self.net(x)