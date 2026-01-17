import torch
import torch.nn as nn
from .embeddings import FourierRepresentation, HashRepresentation
from .decoders import StandardMLP, NeRFDecoder, InstantNeRFDecoder, DeformationNetwork

class NeuralField(nn.Module):
    """神经场模型：坐标 → 编码 → 解码 → 输出"""
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        
        # Part 3 专用：坐标噪声增强配置
        self.use_coord_noise = config.get('use_coord_noise', False)
        self.coord_noise_std = config.get('coord_noise_std', 0.005)
        self.time_noise_std = config.get('time_noise_std', 0.02)
        
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
        elif self.mode == 'part3':
            # 方向编码
            L_dir = config.get('L_embed_dir', 4)
            self.dir_representation = FourierRepresentation(
                input_dim=3, L=L_dir, use_encoding=True
            )
            # 时间编码
            L_time = config.get('L_embed_time', 10)
            self.time_encoder = FourierRepresentation(input_dim=1, L=L_time, use_encoding=True)
            
            # 变形网络：使用与规范网络输入相同的 L_embed 来编码用于变形的位置
            L_pos_for_deform = config.get('L_embed', 10)
            self.pos_encoder_for_deform = FourierRepresentation(input_dim=3, L=L_pos_for_deform, use_encoding=True)
            
            self.deform_net = DeformationNetwork(
                pos_dim=self.pos_encoder_for_deform.out_dim,
                time_dim=self.time_encoder.out_dim,
                hidden_dim=config.get('deform_hidden_dim', 128),
                num_layers=config.get('deform_num_layers', 4)
            )
            
            # 规范网络
            canonical_type = config.get('canonical_type', 'nerf') # 'nerf' or 'instant'
            if canonical_type == 'instant':
                # 使用 Hash Grid 作为规范空间的表示
                self.canonical_repr = HashRepresentation(
                    n_levels=config.get('n_levels', 16),
                    n_features_per_level=config.get('n_features_per_level', 2),
                    log2_hashmap_size=config.get('log2_hashmap_size', 19),
                    base_resolution=config.get('base_resolution', 16),
                    per_level_scale=config.get('per_level_scale', 1.5),
                    bound=config.get('scene_bound', 1.0)
                )
                # 将时间特征拼接到规范空间特征中
                self.decoder = InstantNeRFDecoder(
                    pos_dim=self.canonical_repr.out_dim + self.time_encoder.out_dim,
                    dir_dim=self.dir_representation.out_dim,
                    hidden_dim=config.get('hidden_dim', 64)
                )
            else: # canonical_type == 'nerf'
                # 使用 Fourier MLP 作为规范空间的表示
                L_canon = config.get('L_embed_canon', 10)
                self.canonical_repr = FourierRepresentation(input_dim=3, L=L_canon, use_encoding=True)
                # 将时间特征拼接到规范空间特征中
                self.decoder = NeRFDecoder(
                    pos_dim=self.canonical_repr.out_dim + self.time_encoder.out_dim,
                    dir_dim=self.dir_representation.out_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 8),
                    skip_layer=config.get('skip_layer', 4),
                    view_dim=config.get('view_dim', 128),
                )

            self.direct_time_conditioning = config.get('direct_time_conditioning', False)
            if self.direct_time_conditioning:
                # 使用 Fourier 编码原始位置 (x)
                L_pos = config.get('L_embed', 10)
                self.pos_encoder_direct = FourierRepresentation(input_dim=3, L=L_pos, use_encoding=True)
                # 解码器：输入 = [pos_enc(x), time_enc(t), dir_enc(d)]
                self.decoder_direct = NeRFDecoder(
                    pos_dim=self.pos_encoder_direct.out_dim + self.time_encoder.out_dim,
                    dir_dim=self.dir_representation.out_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 8),
                    skip_layer=config.get('skip_layer', 4),
                    view_dim=config.get('view_dim', 128),
                )

    def forward(self, x, d=None, t=None):
        """
        x: 位置坐标 [N, 2/3]
        d: 视角方向 [N, 3] (仅 part2_nerf/part2_instant)
        t: 时间戳 [N, 1] (仅 part3)
        """
        if self.mode == "part3":
            if t is None:
                raise ValueError("Part 3 requires time input 't'.")
            
            if getattr(self, 'direct_time_conditioning', False):
                # 直接对原始输入进行编码
                feat_x = self.pos_encoder_direct(x)      # embed(x)
                feat_t = self.time_encoder(t)            # embed(t)
                feat_d = self.dir_representation(d)      # embed(d)
                # 拼接位置和时间特征
                h_combined = torch.cat([feat_x, feat_t], dim=-1)
                rgb, sigma = self.decoder_direct(h_combined, feat_d)
                # 此模式无变形场，返回 delta_x 为零
                delta_x = torch.zeros_like(x)
                return rgb, sigma, delta_x

            # ======== A. 坐标噪声增强 ========
            # 在训练阶段给 DeformNet 的输入注入微小噪声
            # 强制模型在 x±ε 范围内输出相似位移，增强变形场平滑性
            x_deform = x
            t_deform = t
            if self.training and self.use_coord_noise:
                # 坐标噪声: x' = x + N(0, σ_coord)
                if self.coord_noise_std > 0:
                    x_deform = x + torch.randn_like(x) * self.coord_noise_std
                # 时间噪声: t' = t + N(0, σ_time)
                if self.time_noise_std > 0:
                    t_deform = t + torch.randn_like(t) * self.time_noise_std
                    # 时间归一化到 [0, 1]，需要 clamp 防止越界
                    t_deform = torch.clamp(t_deform, 0.0, 1.0)
            
            # 变形阶段 - 使用可能带噪声的输入
            feat_t = self.time_encoder(t_deform)              # embed(t')
            feat_x = self.pos_encoder_for_deform(x_deform)    # embed(x') for deformation
            delta_x = self.deform_net(feat_x, feat_t)         # DeformationNet(feat_x', feat_t')
            x_canonical = x + delta_x                         # x_canonical = x + Δx (注意：这里用原始 x)

            # 规范渲染阶段
            feat_can = self.canonical_repr(x_canonical)       # embed_canonical(x_canonical)
            feat_d = self.dir_representation(d)               # embed_dir(d)
            
            # 将时间特征拼接到规范空间特征中，增强时变光影处理能力
            # feat_combined = [x_can_features, time_features]
            h_combined = torch.cat([feat_can, feat_t], dim=-1)
            rgb, sigma = self.decoder(h_combined, feat_d)     # CanonicalNet(feat_combined, feat_d)

            # 返回 rgb, sigma, Δx
            return rgb, sigma, delta_x

        elif self.mode in ["part2_nerf", "part2_instant"]:
            if d is None:
                raise ValueError(f"{self.mode} requires view directions.")
            h = self.representation(x)  # 位置编码
            d = self.dir_representation(d)  # 方向编码
            return self.decoder(h, d)  # → (RGB, σ)

        else:  # part1_fourier
            h = self.representation(x)
            return self.decoder(h)  # → RGB
