import torch

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


def volume_render(rgb, sigma, z_vals, rays_d, white_bkgd):
    """
    体渲染：根据 RGB 和密度计算最终像素颜色
    公式: C(r) = Σ T_i * (1 - exp(-σ_i * δ_i)) * c_i
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

    # 白色背景合成
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]

    return rgb_map, depth_map, acc_map


def render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    perturb,
    white_bkgd,
):
    """
    渲染一批光线
    流程: 采样 3D 点 → NeRF 模型预测 → 体渲染
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]

    # 沿光线采样点
    z_vals = sample_stratified(near, far, n_samples, n_rays, device, perturb)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]  

    # 归一化视角方向
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[:, None, :].expand(-1, n_samples, -1)

    # 展平并输入模型
    pts_flat = pts.reshape(-1, 3)
    view_dirs_flat = view_dirs.reshape(-1, 3)
    rgb, sigma = model(pts_flat, view_dirs_flat)
    
    # 恢复形状
    rgb = rgb.view(n_rays, n_samples, 3)
    sigma = sigma.view(n_rays, n_samples)

    return volume_render(rgb, sigma, z_vals, rays_d, white_bkgd)


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
