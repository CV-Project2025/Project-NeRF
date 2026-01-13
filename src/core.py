import torch.nn as nn
from .embeddings import FourierRepresentation, HashRepresentation
from .decoders import StandardMLP, NeRFDecoder, InstantNeRFDecoder

class NeuralField(nn.Module):
    """神经场模型：坐标 → 编码 → 解码 → 输出"""
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        
        # Part 1 和 Part 2 标准模式需要的参数
        use_pe = config.get('use_positional_encoding', True)
        L = config.get('L_embed', 0) if use_pe else 0
        
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
        # Part 2 Instant: Instant-NeRF
        elif self.mode == 'part2_instant':
            # 哈希编码: 3D 位置 → 特征
            self.representation = HashRepresentation(
                n_levels=config.get('n_levels', 16),
                n_features_per_level=config.get('n_features_per_level', 2),
                log2_hashmap_size=config.get('log2_hashmap_size', 19),
                base_resolution=config.get('base_resolution', 16),
                per_level_scale=config.get('per_level_scale', 1.5),
                bound=config.get('scene_bound', 1.0)
            )
            # 方向编码: 使用 Fourier 编码
            L_dir = config.get('L_embed_dir', 4)
            self.dir_representation = FourierRepresentation(
                input_dim=3, L=L_dir, use_encoding=True
            )
            # Instant-NeRF 解码器: 极简 MLP
            self.decoder = InstantNeRFDecoder(
                pos_dim=self.representation.out_dim,
                dir_dim=self.dir_representation.out_dim,
                hidden_dim=config.get('hidden_dim', 64)
            )

    def forward(self, x, d=None):
        """
        x: 位置坐标 [N, 2/3]
        d: 视角方向 [N, 3] (仅 part2_nerf/part2_instant)
        """
        if self.mode in ["part2_nerf", "part2_instant"]:
            if d is None:
                raise ValueError(f"{self.mode} requires view directions.")
            h = self.representation(x)  # 位置编码
            d = self.dir_representation(d)  # 方向编码
            return self.decoder(h, d)  # → (RGB, σ)

        h = self.representation(x)
        return self.decoder(h)  # → RGB
