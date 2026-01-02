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
            self.freq_bands = 2. ** torch.linspace(0., L - 1, steps=L)
            self._out_dim = input_dim + 2 * input_dim * L
        else:
            self.freq_bands = None
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

class OctreeRepresentation(BaseRepresentation):
    """Part 3: 线性八叉树表达 (Explicit) - 预留接口"""
    def __init__(self, depth=8, feature_dim=32):
        super().__init__()
        self._out_dim = feature_dim
        # TODO: self.octree = ocnn.Octree(depth)

    def forward(self, x):
        # TODO: codes = ocnn.xyz2morton(x)
        # TODO: feats = ocnn.octree_query(self.octree, codes)
        raise NotImplementedError("To be implemented in Part 3 with O-CNN")

    @property
    def out_dim(self):
        return self._out_dim