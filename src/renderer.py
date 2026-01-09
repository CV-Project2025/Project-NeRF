import torch


def sample_stratified(near, far, n_samples, n_rays, device, perturb):
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(n_rays, n_samples)

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals


def volume_render(rgb, sigma, z_vals, rays_d, white_bkgd):
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * dists)
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

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
    device = rays_o.device
    n_rays = rays_o.shape[0]

    z_vals = sample_stratified(near, far, n_samples, n_rays, device, perturb)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]

    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[:, None, :].expand(-1, n_samples, -1)

    pts_flat = pts.reshape(-1, 3)
    view_dirs_flat = view_dirs.reshape(-1, 3)

    rgb, sigma = model(pts_flat, view_dirs_flat)
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
    h, w = rays_o.shape[:2]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgb_chunks = []
    for i in range(0, rays_o.shape[0], chunk):
        rgb_map, _, _ = render_rays(
            model=model,
            rays_o=rays_o[i : i + chunk],
            rays_d=rays_d[i : i + chunk],
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=False,
            white_bkgd=white_bkgd,
        )
        rgb_chunks.append(rgb_map)

    rgb_map = torch.cat(rgb_chunks, dim=0)
    return rgb_map.view(h, w, 3)
