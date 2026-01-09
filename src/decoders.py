import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract import BaseDecoder

class StandardMLP(BaseDecoder):
    """标准 MLP，用于 Part 1"""
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


class NeRFDecoder(BaseDecoder):
    """NeRF MLP with view-dependent color."""
    def __init__(
        self,
        pos_dim,
        dir_dim,
        hidden_dim=256,
        num_layers=8,
        skip_layer=4,
        view_dim=128,
    ):
        super().__init__()
        self.skip_layer = skip_layer

        pts_layers = []
        for i in range(num_layers):
            in_dim = pos_dim if i == 0 else hidden_dim
            if i == skip_layer:
                in_dim += pos_dim
            pts_layers.append(nn.Linear(in_dim, hidden_dim))
        self.pts_layers = nn.ModuleList(pts_layers)

        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.view_layer = nn.Linear(hidden_dim + dir_dim, view_dim)
        self.rgb_layer = nn.Linear(view_dim, 3)

    def forward(self, x, d):
        h = x
        for i, layer in enumerate(self.pts_layers):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = F.relu(layer(h))

        sigma = F.relu(self.sigma_layer(h))
        feat = self.feature_layer(h)

        h = torch.cat([feat, d], dim=-1)
        h = F.relu(self.view_layer(h))
        rgb = torch.sigmoid(self.rgb_layer(h))
        return rgb, sigma
