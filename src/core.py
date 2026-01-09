import torch.nn as nn
from .embeddings import FourierRepresentation, OctreeRepresentation
from .decoders import StandardMLP, NeRFDecoder

class NeuralField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        
        # 读取位置编码配置
        use_pe = config.get('use_positional_encoding', True)
        L = config['L_embed'] if use_pe else 0
        
        # 1. 组装 Representation (策略模式)
        if self.mode == 'part1_fourier':
            self.representation = FourierRepresentation(
                input_dim=2, L=L, use_encoding=use_pe
            )
            self.decoder = StandardMLP(
                input_dim=self.representation.out_dim,
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config.get('num_layers', 3)
            )
        elif self.mode == 'part2_nerf':
            self.representation = FourierRepresentation(
                input_dim=3, L=L, use_encoding=use_pe
            )
            use_dir = config.get("use_viewdirs", True)
            L_dir = config.get("L_embed_dir", 4) if use_dir else 0
            self.dir_representation = FourierRepresentation(
                input_dim=3, L=L_dir, use_encoding=use_dir
            )
            self.decoder = NeRFDecoder(
                pos_dim=self.representation.out_dim,
                dir_dim=self.dir_representation.out_dim,
                hidden_dim=config.get("hidden_dim", 256),
                num_layers=config.get("num_layers", 8),
                skip_layer=config.get("skip_layer", 4),
                view_dim=config.get("view_dim", 128),
            )
        elif self.mode == 'part3_octree':
            self.representation = OctreeRepresentation()
            self.decoder = StandardMLP(
                input_dim=self.representation.out_dim,
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config.get('num_layers', 3)
            )

    def forward(self, x, d=None):
        # 数据流：坐标 -> [表达层] -> 特征 -> [解码层] -> 输出
        if self.mode == "part2_nerf":
            if d is None:
                raise ValueError("part2_nerf requires view directions.")
            h = self.representation(x)
            d = self.dir_representation(d)
            return self.decoder(h, d)

        h = self.representation(x)
        return self.decoder(h)
