import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract import BaseDecoder

class StandardMLP(BaseDecoder):
    """标准 MLP，用于 Part 1 的 2D 图像拟合"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层: 使用 Sigmoid 将输出限制在 [0, 1] (RGB 范围)
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeRFDecoder(BaseDecoder):
    """
    NeRF 解码器：支持视角依赖的颜色渲染
    
    架构:
        位置 x → [MLP + skip] → 密度 σ + 特征
        特征 + 方向 d → [MLP] → RGB
    """
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

        # 位置编码分支: 提取几何特征
        pts_layers = []
        for i in range(num_layers):
            in_dim = pos_dim if i == 0 else hidden_dim
            if i == skip_layer:
                in_dim += pos_dim   # Skip connection：加长宽度
            pts_layers.append(nn.Linear(in_dim, hidden_dim))
        self.pts_layers = nn.ModuleList(pts_layers)

        # 密度头: 几何特征 → 密度 σ (与视角无关)
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        
        # 缓冲层: 几何特征 → 特征 (用于颜色预测)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # 视角依赖分支: 特征 + 视角 → RGB 
        self.view_layer = nn.Linear(hidden_dim + dir_dim, view_dim)
        self.rgb_layer = nn.Linear(view_dim, 3)

    def forward(self, x, d):
        # 位置处理: 通过多层 MLP 提取几何特征
        h = x
        for i, layer in enumerate(self.pts_layers):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)  # Skip connection
            h = F.relu(layer(h))

        # 预测密度 (与视角无关，只依赖位置)
        sigma = F.relu(self.sigma_layer(h))
        
        # 缓冲
        feat = self.feature_layer(h)

        # 颜色处理: 结合几何特征和视角方向
        h = torch.cat([feat, d], dim=-1)
        h = F.relu(self.view_layer(h))
        rgb = torch.sigmoid(self.rgb_layer(h))  # 限制到 [0, 1]
        
        return rgb, sigma
