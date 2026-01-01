import torch.nn as nn
from abc import ABC, abstractmethod

class BaseRepresentation(nn.Module, ABC):
    """
    抽象基类：空间表达层。
    定义了 input(coords) -> output(features) 的标准协议。
    """
    @abstractmethod
    def forward(self, x):
        pass
    
    @property
    @abstractmethod
    def out_dim(self):
        """告诉 Decoder 输入维度是多少，实现动态适配"""
        pass

class BaseDecoder(nn.Module, ABC):
    """
    抽象基类：解码层。
    定义了 input(features) -> output(RGB/Sigma) 的标准协议。
    """
    @abstractmethod
    def forward(self, x):
        pass