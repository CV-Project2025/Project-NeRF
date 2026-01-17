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


class InstantNeRFDecoder(BaseDecoder):
    """
    Instant-NGP 解码器：针对哈希特征优化的极简 MLP
    
    架构设计：
    - 密度支路：1 层隐藏层
    - 颜色支路：2 层隐藏层
    - 使用 FullyFusedMLP（硬件加速，GPU 核函数融合）
    
    """
    def __init__(self, 
                 pos_dim,         # 位置编码维度（来自 HashRepresentation）
                 dir_dim,         # 方向编码维度（通常仍用 Fourier）
                 hidden_dim=64):  # 隐藏层维度（通常用 64）
        super().__init__()
        

        import tinycudann as tcnn
        
        # 密度支路：位置特征 → 密度 + 几何特征
        # 输出：1 维密度 + 15 维几何特征（用于颜色预测）
        self.sigma_net = tcnn.Network(
            n_input_dims=pos_dim,
            n_output_dims=16,  # 1 (sigma) + 15 (geo_features)
            network_config={
                "otype": "FullyFusedMLP",       # TCNN硬件加速
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,           
            }
        )

        # 颜色支路：几何特征 + 视角方向 → RGB
        self.color_net = tcnn.Network(
            n_input_dims=16 + dir_dim,  # geo_features (16) + view_direction
            n_output_dims=3,            # RGB
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",  # 直接输出 [0, 1] 范围
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,           
            }
        )

    def forward(self, x_enc, d_enc):
        """
        前向传播：从编码特征预测 RGB 和密度
        
        Args:
            x_enc: [N, pos_dim] 位置哈希编码
            d_enc: [N, dir_dim] 方向编码
        
        Returns:
            rgb: [N, 3] 颜色
            sigma: [N, 1] 密度
        """
        # 1. 预测密度和几何特征
        h = self.sigma_net(x_enc)  # [N, 16]
        
        # softplus(x) = log(1 + exp(x))，更平滑且梯度更稳定
        # 添加 - 5.0 偏置，让默认密度更低
        sigma = F.softplus(h[..., 0:1] - 5.0) 
        
        # 几何特征用于颜色预测（包含密度信息）
        geo_feat = h  # [N, 16]

        # 2. 预测颜色（结合几何特征和视角）
        color_input = torch.cat([geo_feat, d_enc], dim=-1)  # [N, 16 + dir_dim]
        rgb = self.color_net(color_input)  # [N, 3], 已经是 [0, 1] 范围
        
        return rgb, sigma


class DeformationNetwork(BaseDecoder):
    """
    变形网络：预测从当前时空点 (x, t) 到规范空间的位移 Δx。
    输入: 位置 x (3D) 和 时间 t (1D) 的嵌入特征。
    输出: 位移 Δx (3D)。
    """
    def __init__(self, pos_dim, time_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # 第一层
        layers = [nn.Linear(pos_dim + time_dim, hidden_dim), nn.ReLU()]
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        # 输出层
        output_layer = nn.Linear(hidden_dim, 3)
        # 初始化输出层权重为极小值，确保初始位移接近0
        nn.init.uniform_(output_layer.weight, -1e-4, 1e-4)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x_feat, t_feat):
        # 拼接位置和时间特征
        h = torch.cat([x_feat, t_feat], dim=-1)
        delta_x = self.net(h)
        return delta_x


class DirectTimeDecoder(BaseDecoder):
    """
    直接时间拼接：将 Fourier 编码后的位置、时间和方向特征直接拼接，输入到一个 MLP 中。
    """
    def __init__(
        self,
        pos_dim,      # 位置编码维度
        time_dim,     # 时间编码维度
        dir_dim,      # 方向编码维度
        hidden_dim=256,
        num_layers=8,
        skip_layer=4,
        output_dim=4, # RGB (3) + Sigma (1)
    ):
        super().__init__()
        self.skip_layer = skip_layer
        self.pos_dim = pos_dim
        self.time_dim = time_dim
        self.dir_dim = dir_dim
        
        # 总输入维度 = pos + time + dir
        input_dim = pos_dim + time_dim + dir_dim
        
        # 构建MLP
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
                if i == skip_layer:
                    in_dim += input_dim  # Skip connection
            
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.ModuleList(layers)
    
    def forward(self, x_enc, t_enc, d_enc):
        """
        Args:
            x_enc: [N, pos_dim] 位置编码
            t_enc: [N, time_dim] 时间编码
            d_enc: [N, dir_dim] 方向编码
        Returns:
            rgb: [N, 3]
            sigma: [N, 1]
        """
        # 直接拼接所有特征
        h = torch.cat([x_enc, t_enc, d_enc], dim=-1)
        x_input = h.clone()  # 用于skip connection
        
        # 前向传播
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i == self.skip_layer * 2:  # 因为有ReLU，所以索引*2
                    h = torch.cat([h, x_input], dim=-1)
            h = layer(h)
        
        rgb = torch.sigmoid(h[..., :3])  # [N, 3]
        sigma = F.relu(h[..., 3:4])      # [N, 1]
        return rgb, sigma


class HashDeformationDecoder(BaseDecoder):
    """
    哈希位移解码器

    将哈希网格特征解码为 3D 位移向量 Δx。
    
    核心设计
    - 乘法门控：时间调制通过 element-wise 乘法作用于空间特征
    - 零偏置锚定：t=0 时调制因子接近 0，确保规范空间无变形
    - 极小初始化：displacement_scale 初始值极小，防止训练初期位移过大
    """
    def __init__(self, 
                 hash_dim,           # 位移哈希网格特征维度
                 time_mod_dim,       # 时间调制向量维度
                 hidden_dim=64):     # 隐藏层维度
        super().__init__()
        
        import tinycudann as tcnn
        
        # 位移解码器：哈希特征 + 时间调制 → Δx
        # 使用 2 层 MLP 保持轻量
        self.deform_net = tcnn.Network(
            n_input_dims=hash_dim + time_mod_dim,
            n_output_dims=3,  # (dx, dy, dz)
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",  # 位移可正可负
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            }
        )
        
        # 位移缩放因子：初始值 0.1，允许位移场正常学习（过小会导致位移被压制，过大会导致不稳定）
        self.displacement_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, hash_feat, time_mod):
        """
        Args:
            hash_feat: [N, hash_dim] 位移哈希网格特征
            time_mod: [N, time_mod_dim] 时间调制向量
        
        Returns:
            delta_x: [N, 3] 位移向量
        """
        # 拼接哈希特征和时间调制
        h = torch.cat([hash_feat, time_mod], dim=-1)
        
        # 解码为位移
        delta_x = self.deform_net(h)  # [N, 3]
        
        # 应用缩放因子
        delta_x = delta_x * self.displacement_scale
        
        return delta_x


class TimeModulationNetwork(BaseDecoder):
    """
    时间调制网络
    
    将时间编码映射为调制向量，用于控制位移哈希网格的输出。
    实现时空解耦：空间由 HashGrid 处理，时间由轻量 MLP 处理。
    
    核心设计（零偏置锚定）：
    - 输入：时间的 Fourier 编码
    - 输出：时间调制向量（用于调制位移场）
    - t=0 时输出接近零（规范空间无变形）
    - 使用 sigmoid 门控确保输出在 [0, 1] 范围，再中心化到 [-1, 1]
    """
    def __init__(self, 
                 time_dim,           # 时间编码维度
                 output_dim=64,      # 输出调制向量维度
                 hidden_dim=64,      # 隐藏层维度
                 num_layers=2):      # 网络层数
        super().__init__()
        self.output_dim = output_dim
        
        layers = []
        in_dim = time_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.net = nn.Sequential(*layers)
        
        # 初始化策略：
        # 1. 最后一层权重正常初始化，保证不同时刻有不同输出
        # 2. 偏置初始化为 -1.0，这样 sigmoid(output) 初期偏小
        # 3. 训练时网络会学习到正确的时间调制
        nn.init.xavier_uniform_(layers[-1].weight)  # 正常初始化，保证时间敏感性
        nn.init.constant_(layers[-1].bias, -1.0)    # sigmoid(-1) ≈ 0.27，初期偏小
    
    def forward(self, time_feat):
        """
        Args:
            time_feat: [N, time_dim] 时间编码
        
        Returns:
            time_mod: [N, output_dim] 时间调制向量，范围约 [0, 1]
        """
        raw = self.net(time_feat)
        # 使用 sigmoid 门控，确保 t=0 时调制因子接近 0，这样当 t=0 时，位移会被压制到接近零
        return torch.sigmoid(raw)

