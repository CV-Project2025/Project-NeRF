import torch
import torch.nn as nn


class DensityGrid(nn.Module):
    """
    - 维护一个 3D 位图网格，记录空间中哪些区域有物体（sigma > threshold）
    - 训练时跳过所有 sigma ≈ 0 的空区域（Empty Space Skipping）
    - 定期更新网格（每 16 步），适应场景变化
    - 优势：提高采样效率、收敛速度、渲染质量
    """
    def __init__(self, resolution=128, bound=1.0, threshold=0.01):
        """
        Args:
            resolution: 网格分辨率（默认 128³）
            bound: 场景边界 [-bound, bound]
            threshold: 密度阈值，sigma < threshold 视为空区域
        """
        super().__init__()
        self.resolution = resolution
        self.bound = bound
        self.threshold = threshold
        
        # 密度网格：存储每个 voxel 的平均密度
        self.register_buffer("grid", torch.zeros(resolution, resolution, resolution))
        
        # 二值网格：表示 voxel 有没有物体
        # 参数register_buffer 告诉 PyTorch 这个变量是模型的一部分，但它不是需要梯度更新的“参数”。它会随模型一起保存（.pth），并随模型自动移动到 GPU
        self.register_buffer("binary_grid", torch.ones(resolution, resolution, resolution, dtype=torch.bool))
        
        # 网格坐标 [-bound, bound] → [0, resolution-1]
        self.scale = resolution / (2 * bound)
        self.offset = bound

    @torch.no_grad()
    def update(self, model, n_samples=128**3, device='cuda', time=None, decay=1.0):
        """
        更新占据网格：通过询问 model 各点的密度值，更新 grid 和 binary_grid
        part 3 时空并集更新：保留历史轨迹，取最大值
        
        Args:
            model: NeRF 模型
            n_samples: 采样点数（默认采样所有网格中心）
            device: 要求 cuda
            time: 时间参数 [1, 1]（仅 Part 3 动态模型需要，随机采样时刻）
            decay: 衰减系数（默认 1.0 = 永久记忆），只要物体去过的地方永远标记为活跃
        """
        # 生成网格中心点坐标
        x = torch.linspace(-self.bound, self.bound, self.resolution, device=device)
        y = torch.linspace(-self.bound, self.bound, self.resolution, device=device)
        z = torch.linspace(-self.bound, self.bound, self.resolution, device=device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [resolution³, 3]
        
        # 分批查询密度（避免显存溢出）
        batch_size = 2**18  # 256K 点一批
        density_values = []
        
        # 检测模型是否为 Part 3 动态模型
        mode = getattr(model, 'mode', 'unknown')
        
        for i in range(0, pts.shape[0], batch_size):
            batch_pts = pts[i:i + batch_size]
            
            # 使用零方向（密度与方向无关）
            dummy_dirs = torch.zeros_like(batch_pts)
            
            # 查询模型密度
            with torch.no_grad():
                if mode == 'part3':
                    # 动态模型：使用数据集中间时刻的真实时间戳构建网格
                    if time is None:
                        raise ValueError("Part 3 density grid update requires a time parameter from dataset")
                    time_batch = time.expand(batch_pts.shape[0], -1)
                    _, sigma, _ = model(batch_pts, dummy_dirs, t=time_batch)
                else:
                    # 静态模型
                    _, sigma = model(batch_pts, dummy_dirs)
            
            density_values.append(sigma.squeeze(-1))
        
        # 拼接所有密度值
        all_densities = torch.cat(density_values, dim=0)  # [resolution³]
        
        # 重塑为 3D 网格（当前时刻）
        current_grid = all_densities.reshape(self.resolution, self.resolution, self.resolution)
        
        # 严格时空并集 - 只要过去有或现在有，就标为活跃，确保网格覆盖整个运动轨迹
        mode = getattr(model, 'mode', 'unknown')
        if mode == 'part3':
            # 保留历史 + 当前时刻取最大值
            self.grid = torch.maximum(self.grid * decay, current_grid)
        else:
            self.grid = current_grid
        
        # 更新二值网格：密度 > threshold 的区域标记为活跃
        self.binary_grid = (self.grid > self.threshold)
        
        # 统计活跃 voxel 比例
        active_ratio = self.binary_grid.float().mean().item()
        return active_ratio

    def get_active_mask(self, pts):
        """
        检查采样点是否在活跃区域内
        
        Args:
            pts: [N, 3] 世界坐标
        
        Returns:
            mask: [N] bool 张量，True 表示该点在活跃区域
        """
        # 将世界坐标转换为网格索引 [0, resolution-1]
        indices = ((pts + self.offset) * self.scale).long()
        
        # 边界检查
        valid_mask = (
            (indices >= 0).all(dim=-1) & 
            (indices < self.resolution).all(dim=-1)
        )
        
        # 初始化掩码（默认全部非活跃）
        mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        
        # 查询二值网格
        valid_indices = indices[valid_mask]
        if valid_indices.shape[0] > 0:
            grid_values = self.binary_grid[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2]
            ]
            mask[valid_mask] = grid_values
        
        return mask

    def should_update(self, step, update_interval=16, warmup_iters=0):
        """
        判断是否应该更新网格（每 N 步更新一次，warmup 期间不更新）
        
        Args:
            step: 当前训练步数
            update_interval: 更新间隔（默认 16 步）
            warmup_iters: warmup 迭代数（默认 0，即不 warmup）
        
        Returns:
            bool: 是否应该更新
        """
        # warmup 期间不更新网格
        if step < warmup_iters:
            return False
        return step % update_interval == 0


def sample_stratified(near, far, n_samples, n_rays, device, perturb):
    """沿光线在 [near, far] 范围内分层采样 3D 点"""
    # 线性插值生成采样深度
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(n_rays, n_samples)

    # 训练时添加随机扰动，防止过拟合到固定采样位置
    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals


def volume_render(rgb, sigma, z_vals, rays_d, bg_color=None):
    """
    体渲染：根据 RGB 和密度计算最终像素颜色
    公式: C(r) = Σ T_i * (1 - exp(-σ_i * δ_i)) * c_i + (1 - acc) * bg_color
    
    Args:
        bg_color: 背景颜色 [3] 或 [N_rays, 3]，默认为黑色
    """
    # 计算采样点间距
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    # 计算不透明度 α 和透射率 T
    alpha = 1.0 - torch.exp(-sigma * dists)  
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]  
    weights = alpha * trans  

    # 加权求和得到最终结果
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # 颜色
    depth_map = torch.sum(weights * z_vals, dim=-1)  # 深度
    acc_map = torch.sum(weights, dim=-1)  # 累积不透明度

    # 背景合成：C = C_pred + (1 - acc) * bg_color
    if bg_color is not None:
        # bg_color 可以是 [3] 或 [N_rays, 3]
        if bg_color.dim() == 1:
            bg_color = bg_color.unsqueeze(0)  # [1, 3]
        rgb_map = rgb_map + (1.0 - acc_map)[..., None] * bg_color

    return rgb_map, depth_map, acc_map


def render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    perturb,
    density_grid=None,  # Instant-NeRF 使用 density_grid
    times=None,
    white_bkgd=True,  # 向后兼容：是否使用白色背景
    bg_color=None,  # 背景颜色 [3] 或 [N_rays, 3]，优先级高于 white_bkgd
):
    """
    渲染一批光线
    流程: 采样 3D 点 → NeRF 模型预测 → 体渲染
    
    Args:
        density_grid: DensityGrid 实例（可选），将使用空域跳跃优化，只查询活跃区域。
        times: [N_rays, 1] 时间戳（可选），用于时变场景渲染。(仅 part3)
        white_bkgd: 向后兼容参数，True=白色背景，False=黑色背景
        bg_color: 背景颜色 [3] 或 [N_rays, 3]，如果提供则覆盖 white_bkgd
    
    Returns:
        如果 times 为 None (静态): (rgb_map, depth_map, acc_map)
        如果 times 不为 None (动态): (rgb_map, depth_map, acc_map, extras)
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]
    mode = getattr(model, 'mode', 'unknown')
    
    # 向后兼容：将 white_bkgd 转换为 bg_color
    if bg_color is None:
        if white_bkgd:
            bg_color = torch.ones(3, device=device)  # 白色
        else:
            bg_color = torch.zeros(3, device=device)  # 黑色

    # 兼容性检查与时间处理
    if mode == "part3":
        if times is None:
            # 如果是动态模式但未提供时间，默认使用 t=0
            times = torch.zeros((n_rays, 1), device=device)
        # 广播时间戳到每个采样点 [N_rays, N_samples, 1]
        times_broadcast = times.expand(-1, n_samples).unsqueeze(-1)
    else:
        # 静态模式，忽略 times
        times_broadcast = None

    # 沿光线采样点
    z_vals = sample_stratified(near, far, n_samples, n_rays, device, perturb)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]  

    # 归一化视角方向
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[:, None, :].expand(-1, n_samples, -1)

    # 展平并输入模型
    pts_flat = pts.reshape(-1, 3)
    view_dirs_flat = view_dirs.reshape(-1, 3)
    times_flat = times_broadcast.reshape(-1, 1) if times_broadcast is not None else None

    # Density Grid 过滤空区域
    if density_grid is not None:
        # 获取活跃掩码（只查询有物体的区域）
        active_mask = density_grid.get_active_mask(pts_flat)
        
        # 确保至少查询一个点，避免梯度断流
        # 如果 active_mask 全为 False，强制查询第一个点以保持梯度连接
        if not active_mask.any():
            active_mask = active_mask.clone()
            active_mask[0] = True
        
        # 查询活跃点
        if mode == "part3":
            rgb_compact, sigma_compact, delta_x_compact = model(
                pts_flat[active_mask],
                view_dirs_flat[active_mask],
                t=times_flat[active_mask] if times_flat is not None else None
            )
        else:
            rgb_compact, sigma_compact = model(
                pts_flat[active_mask],
                view_dirs_flat[active_mask]
            )
            delta_x_compact = None
        
        # 使用模型输出创建零张量，继承梯度属性，并显式指定 FP32
        rgb = rgb_compact.new_zeros(pts_flat.shape[0], 3, dtype=torch.float32)
        sigma = sigma_compact.new_zeros(pts_flat.shape[0], 1, dtype=torch.float32)
        
        # 填充计算结果（PyTorch 的索引赋值支持梯度传播）
        rgb[active_mask] = rgb_compact.float()
        sigma[active_mask] = sigma_compact.float()
        
        # 处理变形量
        if mode == "part3" and delta_x_compact is not None:
            delta_x_flat = delta_x_compact.new_zeros(pts_flat.shape[0], 3, dtype=torch.float32)
            delta_x_flat[active_mask] = delta_x_compact.float()
        else:
            delta_x_flat = None
        
        rgb = rgb.float().view(n_rays, n_samples, 3)
        sigma = sigma.float().view(n_rays, n_samples)

    else:
        # 标准路径 (适用于所有模式)
        if mode == "part3":
            # 动态模式: 调用带时间参数的模型
            rgb_flat, sigma_flat, delta_x_flat = model(pts_flat, view_dirs_flat, t=times_flat)
        # 标准 NeRF: 查询所有点
        else:
            rgb_flat, sigma_flat = model(pts_flat, view_dirs_flat)
            delta_x_flat = None
    
        # 恢复形状
        rgb = rgb_flat.float().view(n_rays, n_samples, 3)
        sigma = sigma_flat.float().view(n_rays, n_samples)

    # 体渲染（直接传入 bg_color）
    rgb_map, depth_map, acc_map = volume_render(rgb, sigma, z_vals, rays_d, bg_color=bg_color)

    # 动态返回值：根据是否提供 times 参数决定返回格式，保持向后兼容
    if times is not None:
        extras = {}
        if mode == "part3" and delta_x_flat is not None:
            # 从 volume_render 复用逻辑计算 weights
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
            dists = dists * torch.norm(rays_d[:, None, :], dim=-1)
            alpha = 1.0 - torch.exp(-sigma * dists)
            trans = torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1),
                dim=-1,
            )[:, :-1]
            weights = alpha * trans  # [N_rays, N_samples]

            # 恢复 delta_x 的形状并计算加权平均
            delta_x = delta_x_flat.view(n_rays, n_samples, 3)
            mean_delta_x = torch.sum(weights.unsqueeze(-1) * delta_x, dim=1)  # [N_rays, 3]
            extras['mean_delta_x'] = mean_delta_x
        
        return rgb_map, depth_map, acc_map, extras
    else:
        return rgb_map, depth_map, acc_map


def render_image(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    chunk,
    white_bkgd,
):
    """渲染完整图像（分块处理节省显存）"""
    h, w = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    # 分块渲染并拼接
    rgb_chunks = []
    for i in range(0, rays_o.shape[0], chunk):
        rgb_map, _, _ = render_rays(
            model=model,
            rays_o=rays_o[i : i + chunk],
            rays_d=rays_d[i : i + chunk],
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=False,  # 评估时不加扰动
            white_bkgd=white_bkgd,
        )
        rgb_chunks.append(rgb_map)

    rgb_map = torch.cat(rgb_chunks, dim=0)
    return rgb_map.view(h, w, 3)
