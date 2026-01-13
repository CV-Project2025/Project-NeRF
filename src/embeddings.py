import torch
import torch.nn as nn
import numpy as np
from .abstract import BaseRepresentation

class FourierRepresentation(BaseRepresentation):
    """Part 1 & Part 2: 标准位置编码 (Implicit)"""
    def __init__(self, input_dim=2, L=10, use_encoding=True):
        super().__init__()
        self.input_dim = input_dim # Part1=2, Part2=3
        self.L = L
        self.use_encoding = use_encoding
        
        if use_encoding and L > 0:
            freq_bands = 2. ** torch.linspace(0.0, L - 1, steps=L)
            self.register_buffer("freq_bands", freq_bands)
            self._out_dim = input_dim + 2 * input_dim * L
        else:
            self.register_buffer("freq_bands", torch.empty(0))
            self._out_dim = input_dim

    def forward(self, x):
        if not self.use_encoding or self.L == 0:
            # 无编码：直接返回原始坐标
            return x
            
        # 标准 Fourier 位置编码
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq * np.pi))
            encoded.append(torch.cos(x * freq * np.pi))
        return torch.cat(encoded, dim=-1)

    @property
    def out_dim(self):
        return self._out_dim


class HashRepresentation(BaseRepresentation):
    """
    多分辨率哈希编码
    
    使用 tiny-cuda-nn 的哈希网格实现 O(1) 查询复杂度，将 3D 空间划分为多个分辨率的网格，顶点映射到哈希表中。
    
    """
    def __init__(self, 
                 n_levels=16,              # 分辨率层级数
                 n_features_per_level=2,   # 每层特征数
                 log2_hashmap_size=19,     # 哈希表大小 log2(2^19 = 512K)
                 base_resolution=16,       # 最粗层的分辨率
                 per_level_scale=1.5,      # 每层分辨率增长系数
                 bound=1.0):               # 场景边界 [-bound, bound]
        super().__init__()
        self.bound = bound
        

        import tinycudann as tcnn
        
        # 配置 TCNN 哈希编码
        encoding_config = {
            "otype": "HashGrid",                    # 哈希网格编码
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        
        self.encoding = tcnn.Encoding(
            n_input_dims=3,                         # 输入：3D 坐标 (x, y, z)
            encoding_config=encoding_config
        )
        self._out_dim = self.encoding.n_output_dims  # 输出维度 = n_levels * n_features_per_level

    def forward(self, x):
        """
        将 3D 坐标编码为哈希特征
        
        Args:
            x: [N, 3] 世界坐标
        
        Returns:
            features: [N, out_dim] 哈希编码特征
        """
        # tiny-cuda-nn 的要求，将世界坐标 [-bound, bound] 映射到 [0, 1]
        x_normalized = (x + self.bound) / (2 * self.bound)
        x_clamped = x_normalized.clamp(0.0, 1.0)
        
        return self.encoding(x_clamped)

    @property
    def out_dim(self):
        return self._out_dim

