import torch.nn as nn
from .embeddings import FourierRepresentation, OctreeRepresentation
from .decoders import StandardMLP, NeRFDecoder

class NeuralField(nn.Module):
    """神经场模型：坐标 → 编码 → 解码 → 输出"""
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        
        use_pe = config.get('use_positional_encoding', True)
        L = config['L_embed'] if use_pe else 0
        
        # Part 1: 2D 图像拟合
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
        # Part 2: NeRF 3D 场景重建
        elif self.mode == 'part2_nerf':
            # 位置编码: (x, y, z) → 特征
            self.representation = FourierRepresentation(
                input_dim=3, L=L, use_encoding=use_pe
            )
            # 方向编码: (dx, dy, dz) → 特征
            use_dir = config.get("use_viewdirs", True)
            L_dir = config.get("L_embed_dir", 4) if use_dir else 0
            self.dir_representation = FourierRepresentation(
                input_dim=3, L=L_dir, use_encoding=use_dir
            )
            # NeRF 解码器: (位置, 方向) → (RGB, σ)
            self.decoder = NeRFDecoder(
                pos_dim=self.representation.out_dim,
                dir_dim=self.dir_representation.out_dim,
                hidden_dim=config.get("hidden_dim", 256),
                num_layers=config.get("num_layers", 8),
                skip_layer=config.get("skip_layer", 4),
                view_dim=config.get("view_dim", 128),
            )

    def forward(self, x, d=None):
        """
        x: 位置坐标 [N, 2/3]
        d: 视角方向 [N, 3] (仅 part2_nerf)
        """
        if self.mode == "part2_nerf":
            if d is None:
                raise ValueError("part2_nerf requires view directions.")
            h = self.representation(x)  # 位置编码
            d = self.dir_representation(d)  # 方向编码
            return self.decoder(h, d)  # → (RGB, σ)

        h = self.representation(x)
        return self.decoder(h)  # → RGB
