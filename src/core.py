import torch.nn as nn
from .embeddings import FourierRepresentation, OctreeRepresentation
from .decoders import StandardMLP

class NeuralField(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 读取位置编码配置
        use_pe = config.get('use_positional_encoding', True)
        L = config['L_embed'] if use_pe else 0
        
        # 1. 组装 Representation (策略模式)
        if config['mode'] == 'part1_fourier':
            self.representation = FourierRepresentation(
                input_dim=2, L=L, use_encoding=use_pe
            )
        elif config['mode'] == 'part2_nerf':
            self.representation = FourierRepresentation(
                input_dim=3, L=L, use_encoding=use_pe
            )
        elif config['mode'] == 'part3_octree':
            self.representation = OctreeRepresentation()
            
        # 2. 组装 Decoder (自动适配维度)
        self.decoder = StandardMLP(
            input_dim=self.representation.out_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config.get('num_layers', 3)
        )

    def forward(self, x):
        # 数据流：坐标 -> [表达层] -> 特征 -> [解码层] -> 颜色
        h = self.representation(x)
        rgb = self.decoder(h)
        return rgb