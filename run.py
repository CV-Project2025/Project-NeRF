import argparse
import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml

from src.core import NeuralField
from src.dataset import BlenderDataset
from src.renderer import render_image, render_rays
from src.utils import compute_psnr, compute_psnr_torch, render_image_safe, TensorBoardLogger, get_exp_name


def run_part1(cfg, args):
    """Part 1: 2D 图像拟合"""

    # 参数对比相关
    epochs = cfg["epochs"]
    learning_rate = cfg["learning_rate"]
    batch_size = cfg.get("batch_size", None)
    image_size = cfg.get("image_size", 400)
    log_dir = cfg.get("log_dir", "output/")
    
    # 获取图像名称（不含扩展名）并添加到输出路径
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    log_dir = os.path.join(log_dir, "part1", image_name)
    
    save_every = cfg.get("save_every", 500)
    log_every = cfg.get("log_every", 100)  # 日志记录频率
    output_dim = cfg["output_dim"]
    def ensure_list(value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    use_pe_list = ensure_list(cfg.get("use_positional_encoding", True))
    l_embed_list = ensure_list(cfg["L_embed"])
    hidden_dim_list = ensure_list(cfg["hidden_dim"])
    num_layers_list = ensure_list(cfg.get("num_layers", 3))
    param_combos = list(
        itertools.product(use_pe_list, l_embed_list, hidden_dim_list, num_layers_list)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 加载并处理2D图像
    img = Image.open(args.image).convert("RGB")
    w_orig, h_orig = img.size
    scale = min(image_size / w_orig, image_size / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 生成归一化坐标网格和颜色
    img_np = np.array(img) / 255.0
    h, w, _ = img_np.shape
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)
    gt_rgb = torch.tensor(img_np.reshape(-1, 3), dtype=torch.float32)
    coords, gt_rgb = coords.to(device), gt_rgb.to(device)

    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "final_psnr.csv")
    results_exists = os.path.exists(results_path)

    loss_fn = nn.MSELoss()

    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location=device)
        ckpt_cfg = ckpt.get("config", cfg)
        model = NeuralField(ckpt_cfg).to(device)
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f">>> Warning: load_state_dict missing={load_result.missing_keys}, "
                  f"unexpected={load_result.unexpected_keys}")
        model.eval()
        with torch.no_grad():
            pred = model(coords)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss = loss_fn(pred, gt_rgb).item()
            psnr = compute_psnr(loss)
            final_img = pred.cpu().numpy().reshape(h, w, 3)

        eval_dir = os.path.join(log_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(eval_dir, f"{ckpt_name}.png")
        plt.imsave(out_path, final_img)
        print(f">>> Eval PSNR: {psnr:.2f} dB")
        print(f">>> Rendered image saved to: {out_path}")
        return

    total_pixels = coords.shape[0]
    print(">>> Start Training Part 1 (2D Fitting)...")
    print(
        f">>> 图像尺寸: {h}x{w}, 批量大小: {'全图' if batch_size is None else batch_size}"
    )
    print(f">>> 参数组合数: {len(param_combos)}")

    # 初始化 TensorBoard
    tb_base_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_base_dir, exist_ok=True)
    print(f">>> tensorboard --logdir={tb_base_dir} 查看 TensorBoard 日志")
    
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "use_positional_encoding",
            "L_embed",
            "hidden_dim",
            "num_layers",
            "epochs",
            "learning_rate",
            "batch_size",
            "image_size",
            "final_psnr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not results_exists:
            writer.writeheader()

        for run_idx, (use_pe, l_embed, hidden_dim, num_layers) in enumerate(
            param_combos, start=1
        ):
            config = {
                "mode": cfg["mode"],
                "L_embed": l_embed,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "num_layers": num_layers,
                "use_positional_encoding": use_pe,
            }

            run_name = f"pe{int(bool(use_pe))}_L{l_embed}_H{hidden_dim}_N{num_layers}"
            run_dir = os.path.join(log_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            # 初始化 TensorBoard logger
            tb_dir = os.path.join(tb_base_dir, run_name)
            tb_logger = TensorBoardLogger(tb_dir)

            save_intermediate = isinstance(save_every, int) and save_every > 0
            if save_intermediate:
                steps_dir = os.path.join(run_dir, "steps")
                os.makedirs(steps_dir, exist_ok=True)

            print(f">>> [{run_idx}/{len(param_combos)}] 配置: {run_name}, Steps={epochs}")

            model = NeuralField(config).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练循环
            for i in tqdm(range(epochs)):
                if batch_size is None:
                    # 全图训练
                    pred_rgb = model(coords)
                    loss = loss_fn(pred_rgb, gt_rgb)
                else:
                    # 随机批量采样
                    indices = torch.randint(0, total_pixels, (batch_size,), device=device)
                    batch_coords = coords[indices]
                    batch_gt_rgb = gt_rgb[indices]
                    pred_rgb = model(batch_coords)
                    loss = loss_fn(pred_rgb, batch_gt_rgb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录到 TensorBoard
                if (i + 1) % log_every == 0:
                    psnr = compute_psnr(loss.item())
                    tb_logger.log_scalar('Train/Loss', loss.item(), i + 1)
                    tb_logger.log_scalar('Train/PSNR', psnr, i + 1)

                # 定期保存中间结果
                if save_intermediate and (i + 1) % save_every == 0:
                    with torch.no_grad():
                        intermediate_img = model(coords).cpu().numpy().reshape(h, w, 3)
                    plt.imsave(
                        os.path.join(steps_dir, f"step_{i+1:05d}.png"),
                        intermediate_img,
                    )

            # 训练完成，生成最终结果
            with torch.no_grad():
                final_pred = model(coords)
                final_img = final_pred.cpu().numpy().reshape(h, w, 3)
                final_loss = loss_fn(final_pred, gt_rgb).item()

            final_img_path = os.path.join(run_dir, "final.png")
            plt.imsave(final_img_path, final_img)
            model_path = os.path.join(run_dir, "model_final.pth")
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config},
                model_path,
            )

            final_psnr = compute_psnr(final_loss)
            writer.writerow(
                {
                    "use_positional_encoding": use_pe,
                    "L_embed": l_embed,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "final_psnr": final_psnr,
                }
            )
            f.flush()
            
            # 记录最终 PSNR 到 TensorBoard
            tb_logger.log_scalar('Final/PSNR', final_psnr, epochs)
            tb_logger.close()

            print(f">>> Done! Final PSNR: {final_psnr:.2f} dB")


def run_part2(cfg, args):
    """Part 2: NeRF 3D场景重建，训练和评估"""
    if not args.data_dir:
        raise ValueError("Part 2 requires --data_dir pointing to a NeRF dataset root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 读取渲染和训练配置
    downscale = cfg.get("downscale", 1)
    white_bkgd = cfg.get("white_bkgd", True)
    scene_scale = cfg.get("scene_scale", 1.0)
    near = float(cfg.get("near", 2.0))  # 近平面
    far = float(cfg.get("far", 6.0))  # 远平面
    n_samples = cfg.get("n_samples", 64)  # 训练采样点数
    render_n_samples = cfg.get("render_n_samples", n_samples)  # 渲染采样点数
    batch_size = cfg.get("batch_size", 4096)  # 每批光线数
    train_iters = cfg.get("train_iters", 20000)  # 训练迭代数
    learning_rate = cfg.get("learning_rate", 5e-4)
    log_every = cfg.get("log_every", 100)
    save_every = cfg.get("save_every", 2000)
    chunk = cfg.get("chunk", 8192)  # 渲染块大小
    log_dir = cfg.get("log_dir", "output/part2")
    if args.render_chunk:
        chunk = args.render_chunk

    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    # 加载训练和测试数据集
    train_set = BlenderDataset(
        root_dir=args.data_dir,
        split="train",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    test_split = "test"
    test_meta = os.path.join(args.data_dir, "transforms_test.json")
    if not os.path.exists(test_meta):
        test_split = "val"
    test_set = BlenderDataset(
        root_dir=args.data_dir,
        split=test_split,
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )

    # 初始化模型
    model = NeuralField(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f">>> Loaded checkpoint: {args.checkpoint}")

    # 训练阶段
    if not args.eval_only:
        # 初始化 TensorBoard
        tb_dir = os.path.join(log_dir, "tensorboard")
        tb_logger = TensorBoardLogger(tb_dir)
        print(f">>> tensorboard --logdir={tb_dir} 查看 TensorBoard 日志")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        print(">>> Start Training Part 2 (NeRF)...")
        model.train()
        for step in range(1, train_iters + 1):
            # 随机采样光线并渲染
            rays_o, rays_d, target_rgba = train_set.sample_random_rays(batch_size, device)
            
            # 分离并合成 target (使用固定背景)
            target_rgb = target_rgba[:, :3]
            target_alpha = target_rgba[:, 3:4]
            if white_bkgd:
                target = target_rgb * target_alpha + (1.0 - target_alpha)
            else:
                target = target_rgb * target_alpha
            
            pred_rgb, _, _ = render_rays(
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
                perturb=True,
                white_bkgd=white_bkgd,
            )
            loss = loss_fn(pred_rgb, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                psnr = compute_psnr(loss.item())
                print(
                    f">>> Step {step}/{train_iters} | Loss {loss.item():.6f} | PSNR {psnr:.2f} dB"
                )
                
                # 记录到 TensorBoard
                tb_logger.log_scalar('Train/Loss', loss.item(), step)
                tb_logger.log_scalar('Train/PSNR', psnr, step)

            if save_every and step % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"model_step_{step:06d}.pth")
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path
                )

        final_path = os.path.join(ckpt_dir, "model_final.pth")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, final_path)
        
        tb_logger.close()
        print(f">>> 训练完成！TensorBoard 日志已保存到: {tb_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 评估阶段：渲染测试集
    model.eval()
    print(f">>> Rendering {test_split} set...")
    psnrs = []
    with torch.no_grad():
        for idx in range(len(test_set)):
            rays_o, rays_d, target = test_set.get_image_rays(idx, device)
            pred = render_image_safe(
                render_fn=render_image,
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=render_n_samples,
                chunk=chunk,
                white_bkgd=white_bkgd,
            )
            pred = torch.clamp(pred, 0.0, 1.0)
            psnr = compute_psnr_torch(pred, target)
            psnrs.append(psnr)
            plt.imsave(
                os.path.join(render_dir, f"test_{idx:03d}.png"),
                pred.cpu().numpy(),
            )

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    print(f">>> Test PSNR: {avg_psnr:.2f} dB")
    print(f">>> Rendered images saved to: {render_dir}")


def run_part2_instant(cfg, args):
    """Part 2 Instant: Instant-NeRF 加速训练"""
    if not args.data_dir:
        raise ValueError("Part 2 Instant requires --data_dir pointing to a NeRF dataset root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")
    
    if device.type != 'cuda':
        print("!!! Instant-NeRF 在 CPU 上性能无法发挥，强烈建议使用 CUDA GPU")

    # 读取渲染和训练配置
    downscale = cfg.get("downscale", 2)
    white_bkgd = cfg.get("white_bkgd", True)
    scene_scale = cfg.get("scene_scale", 1.0)
    near = float(cfg.get("near", 2.0))
    far = float(cfg.get("far", 6.0))
    n_samples = cfg.get("n_samples", 32)  # Instant-NeRF 需要更少采样点
    render_n_samples = cfg.get("render_n_samples", n_samples)
    batch_size = cfg.get("batch_size", 8192)  # Instant-NeRF 使用更大批量
    train_iters = cfg.get("train_iters", 5000)  # Instant-NeRF 训练更快
    learning_rate = cfg.get("learning_rate", 0.01)  # Instant-NeRF 使用高学习率
    log_every = cfg.get("log_every", 50)
    save_every = cfg.get("save_every", 500)
    chunk = cfg.get("chunk", 16384)  # 更大的渲染块
    log_dir = cfg.get("log_dir", "output/part2_instant")
    
    # 获取数据集名称并添加到输出路径
    dataset_name = os.path.basename(args.data_dir)
    log_dir = os.path.join(log_dir, dataset_name)
    
    if args.render_chunk:
        chunk = args.render_chunk

    # Instant-NeRF 特有配置
    use_density_grid = cfg.get("use_density_grid", True)
    grid_resolution = cfg.get("grid_resolution", 128)
    grid_threshold = cfg.get("grid_threshold", 0.01)
    grid_update_interval = cfg.get("grid_update_interval", 16)
    grid_warmup_iters = cfg.get("grid_warmup_iters", 256)

    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir)
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    # 加载训练和测试数据集
    train_set = BlenderDataset(
        root_dir=args.data_dir,
        split="train",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    # 加载测试集
    test_split = "test"
    test_meta = os.path.join(args.data_dir, "transforms_test.json")
    if not os.path.exists(test_meta):
        test_split = "val"
    test_set = BlenderDataset(
        root_dir=args.data_dir,
        split=test_split,
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )

    
    # 只在训练模式下划分验证集
    if not args.eval_only:
        # 从测试集中随机抽取30%作为验证集
        import random
        n_test = len(test_set.images)
        n_val = int(n_test * 0.3)
        val_indices = random.sample(range(n_test), n_val)
        test_indices = [i for i in range(n_test) if i not in val_indices]
        
        # 创建验证集
        val_set = BlenderDataset(
            root_dir=args.data_dir,
            split=test_split,
            downscale=downscale,
            white_bkgd=white_bkgd,
            scene_scale=scene_scale,
        )
        val_set.images = test_set.images[val_indices]
        val_set.poses = test_set.poses[val_indices]
        
        # 不缩减测试集，保持完整
        print(f">>> 数据集划分: 训练集 {len(train_set.images)} 张 | 验证集 {len(val_set.images)} 张 | 测试集 {len(test_set.images)} 张")
    else:
        # 评估模式：使用全部测试集
        print(f">>> 评估使用全部测试集 {len(test_set.images)} 张")
        val_set = None

    # 初始化模型
    print(">>> 初始化 Instant-NeRF 模型...")
    model = NeuralField(cfg).to(device)
    

    # 自动检测 scene_bound（如果配置中设置为 "auto"）
    if cfg.get('scene_bound') == 'auto':
        # 从训练集和测试集姿态中提取相机位置
        all_poses = torch.cat([train_set.poses, test_set.poses], dim=0)
        cam_positions = all_poses[:, :3, 3].cpu().numpy()
        
        # 计算相机到原点的最大距离
        max_distance = np.max(np.linalg.norm(cam_positions, axis=1))
        
        # 添加5%余量作为 scene_bound
        scene_bound_auto = max_distance * 1.05
        cfg['scene_bound'] = scene_bound_auto
        print(f">>> 自动检测 scene_bound: {scene_bound_auto:.2f}（基于相机最大距离 {max_distance:.2f}）")


    # 初始化占据网格
    density_grid = None
    active_ratio = 1.0  # 初始化活跃比例（warmup 期间默认 100%）
    if use_density_grid:
        from src.renderer import DensityGrid
        density_grid = DensityGrid(
            resolution=grid_resolution,
            bound=cfg.get('scene_bound', 1.5),
            threshold=grid_threshold
        ).to(device)
        print(f">>> Density Grid 已启用: {grid_resolution}³ 分辨率")
    else:
        print(">>> Density Grid 已禁用（性能会降低）")
    
    # 加载检查点（包括 density_grid）
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if density_grid is not None and "density_grid" in ckpt:
            density_grid.load_state_dict(ckpt["density_grid"])
            print(f">>> Loaded checkpoint with DensityGrid: {args.checkpoint} (Step {ckpt.get('step', '未知')} | Val PSNR {ckpt.get('val_psnr', None):.2f} dB)")
        else:
            print(f">>> Loaded checkpoint: {args.checkpoint} (Step {ckpt.get('step', '未知')} | Val PSNR {ckpt.get('val_psnr', None):.2f} dB)")

    # 训练阶段
    if not args.eval_only:
        # 初始化 TensorBoard
        tb_dir = os.path.join(log_dir, "tensorboard", get_exp_name(cfg))
        tb_logger = TensorBoardLogger(tb_dir)
        
        # AdamW 优化器
        weight_decay = cfg.get('weight_decay', 1e-5)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Cosine 衰减调度器
        eta_min = cfg.get('eta_min', 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=eta_min)
        
        # 随机背景增强
        use_random_bg = cfg.get('use_random_bg', False)
        
        # TV Loss (Total Variation) - 惩罚 HashGrid 相邻特征差异，消除边缘毛刺
        use_tv_loss = cfg.get('use_tv_loss', True)
        tv_loss_weight = float(cfg.get('tv_loss_weight', 1e-6))
        
        loss_fn = nn.MSELoss()

        print(f">>> 目标: {train_iters} 步")
        print(f">>> 学习率: {learning_rate} (Cosine 衰减至 {eta_min})")
        print(f">>> 批量大小: {batch_size}")
        print(f">>> 采样点数: {n_samples} ")
        if use_tv_loss:
            print(f">>> 正则化: TV Loss 已启用 (weight={tv_loss_weight:.0e})")
        if use_random_bg:
            random_bg_start = cfg.get('random_bg_start', 0)
            if random_bg_start > 0:
                print(f">>> 数据增强: 随机背景增强 ({random_bg_start} 步后开启)")
            else:
                print(f">>> 数据增强: 随机背景增强 (全程启用)")
        print(f">>> tensorboard --logdir={os.path.join(log_dir, 'tensorboard')} 查看 TensorBoard 日志")
        
        # 初始化最佳验证集PSNR跟踪
        best_val_psnr = 0.0
        
        model.train()
        for step in range(1, train_iters + 1):
            # 随机采样光线并渲染 (返回 RGBA 4通道)
            rays_o, rays_d, target_rgba = train_set.sample_random_rays(batch_size, device)
            
            # 分离 RGB 和 Alpha 通道
            target_rgb = target_rgba[:, :3]    # [B, 3]
            target_alpha = target_rgba[:, 3:4] # [B, 1]
            
            # 随机背景增强：从 random_bg_start 步开始启用
            if use_random_bg and step >= random_bg_start:
                bg_color = torch.rand(3, device=device)
            else:
                bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
            
            # 动态合成 target: RGB * Alpha + bg_color * (1 - Alpha)
            target = target_rgb * target_alpha + bg_color * (1.0 - target_alpha)
            
            # 使用占据网格加速渲染
            pred_rgb, _, _ = render_rays(
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
                perturb=True,
                white_bkgd=white_bkgd,
                density_grid=density_grid,
                bg_color=bg_color,
            )
            loss_rgb = loss_fn(pred_rgb, target)
            
            # TV Loss - 惩罚 HashGrid 哈希表中相邻条目的特征差异
            loss_tv = torch.tensor(0.0, device=device)
            if use_tv_loss and hasattr(model, 'representation') and hasattr(model.representation, 'encoding'):
                hash_params = model.representation.encoding.params  # [N_entries, n_features]
                tv_diff = torch.abs(hash_params[1:] - hash_params[:-1])  # L1 范数
                loss_tv = torch.mean(tv_diff) * tv_loss_weight
            
            loss = loss_rgb + loss_tv

            optimizer.zero_grad()
            loss.backward()
            
            # 分别裁剪散列表和 MLP 的梯度
            if hasattr(model, 'representation'):
                torch.nn.utils.clip_grad_norm_(model.representation.parameters(), max_norm=1.0)
            if hasattr(model, 'decoder'):
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Cosine 衰减

            # 动态网格更新：前 10% 步数每 16 步更新，10%-50% 每 64 步，50% 后每 256 步
            # 训练后期（grid_stop_ratio 后）停止更新
            grid_stop_ratio = cfg.get('grid_stop_ratio', 0.9)
            if step < train_iters * grid_stop_ratio:
                if step < train_iters * 0.1:
                    dynamic_interval = 32
                elif step < train_iters * 0.5:
                    dynamic_interval = 128
                else:
                    dynamic_interval = 512
                
                if density_grid is not None and density_grid.should_update(step, dynamic_interval, grid_warmup_iters):
                    model.eval()
                    active_ratio = density_grid.update(model, device=device, time=None)
                    model.train()

            # 日志输出和 TensorBoard 记录
            if step % log_every == 0:
                psnr = compute_psnr(loss_rgb.item())
                skip_info = ""
                if density_grid is not None:
                    skip_info = f" | Skip: {(1-active_ratio)*100:.1f}%"
                print(
                    f">>> Step {step}/{train_iters} | Loss {loss.item():.6f} | PSNR {psnr:.2f} dB{skip_info}"
                )
                
                # 记录到 TensorBoard
                tb_logger.log_scalar('Train/Loss', loss_rgb.item(), step)
                tb_logger.log_scalar('Train/PSNR', psnr, step)
                if use_tv_loss:
                    tb_logger.log_scalar('Train/TV_Loss', loss_tv.item(), step)
                if density_grid is not None:
                    tb_logger.log_scalar('Train/ActiveRatio', active_ratio, step)
            
            # 定期验证集评估
            val_every = cfg.get("val_every", 500)
            if step % val_every == 0:
                model.eval()
                val_psnrs = []
                with torch.no_grad():
                    for idx in range(len(val_set.images)):
                        rays_o, rays_d, target = val_set.get_image_rays(idx, device)
                        rays_o = rays_o.reshape(-1, 3)
                        rays_d = rays_d.reshape(-1, 3)
                        target = target.reshape(-1, 3)
                        
                        # 分块渲染验证集
                        pred_chunks = []
                        for i in range(0, rays_o.shape[0], chunk):
                            pred_chunk, _, _ = render_rays(
                                model=model,
                                rays_o=rays_o[i:i+chunk],
                                rays_d=rays_d[i:i+chunk],
                                near=near,
                                far=far,
                                n_samples=render_n_samples,
                                perturb=False,
                                white_bkgd=white_bkgd,
                                density_grid=density_grid,
                            )
                            pred_chunks.append(pred_chunk)
                        pred = torch.cat(pred_chunks, dim=0)
                        val_psnr = compute_psnr_torch(pred, target)
                        val_psnrs.append(val_psnr)
                
                avg_val_psnr = float(np.mean(val_psnrs))
                print(f"    [Validation] PSNR: {avg_val_psnr:.2f} dB", end="")
                
                # 记录验证集 PSNR 到 TensorBoard
                tb_logger.log_scalar('Validation/PSNR', avg_val_psnr, step)
                
                # 只在验证集PSNR提升时保存模型
                if avg_val_psnr > best_val_psnr:
                    best_val_psnr = avg_val_psnr
                    best_path = os.path.join(ckpt_dir, f"best_model.pth")
                    save_dict = {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "val_psnr": best_val_psnr
                    }
                    if density_grid is not None:
                        save_dict["density_grid"] = density_grid.state_dict()
                    torch.save(save_dict, best_path)
                    print(f" | 🌟 New Best Model! Saved to {best_path}")
                else:
                    print()
                
                model.train()

        print(f"\n>>> 训练完成！最佳验证集 PSNR: {best_val_psnr:.2f} dB")
        tb_logger.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 评估阶段
    if args.eval_only:
        import random
        import shutil
        import subprocess
        
        # 判断是否顺序渲染并生成视频（-1 表示全部测试集）
        if args.render_n == -1:
            n_render = len(test_set.images)
            render_indices = list(range(n_render))
            
            # 创建临时图片目录和视频目录
            picture_dir = os.path.join(log_dir, "picture")
            video_dir = os.path.join(log_dir)
            os.makedirs(picture_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            
            print(f"\n>>> 渲染全部测试集图片（按顺序 {n_render} 张）用于生成视频...")
            
            model.eval()
            psnrs = []
            with torch.no_grad():
                for i, idx in enumerate(tqdm(render_indices)):
                    rays_o, rays_d, target = test_set.get_image_rays(idx, device)
                    H, W = rays_o.shape[:2]
                    rays_o = rays_o.reshape(-1, 3)
                    rays_d = rays_d.reshape(-1, 3)
                    
                    # 使用 density_grid 加速渲染
                    pred_chunks = []
                    for j in range(0, rays_o.shape[0], chunk):
                        pred_chunk, _, _ = render_rays(
                            model=model,
                            rays_o=rays_o[j:j+chunk],
                            rays_d=rays_d[j:j+chunk],
                            near=near,
                            far=far,
                            n_samples=render_n_samples,
                            perturb=False,
                            white_bkgd=white_bkgd,
                            density_grid=density_grid,  # 使用占据网格加速
                        )
                        pred_chunks.append(pred_chunk)
                    
                    pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                    pred = torch.clamp(pred, 0.0, 1.0)
                    psnr = compute_psnr_torch(pred, target)
                    psnrs.append(psnr)
                    
                    # 保存为连续编号的帧
                    plt.imsave(
                        os.path.join(picture_dir, f"frame_{i:03d}.png"),
                        pred.cpu().numpy(),
                    )
            
            avg_psnr = float(np.mean(psnrs))
            print(f"\n>>> 渲染完成！平均 PSNR: {avg_psnr:.2f} dB")
            
            # 使用 ffmpeg 生成视频
            dataset_name = os.path.basename(args.data_dir)
            video_path = os.path.join(video_dir, f"{dataset_name}_24fps.mp4")
            print(f"\n>>> 使用 ffmpeg 生成视频...")
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", "24",
                    "-i", os.path.join(picture_dir, "frame_%03d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    video_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f">>> 视频已保存: {video_path}")
                    print(f">>> 视频时长: {n_render/24:.1f}秒 ({n_render} 帧 @ 24fps)")
                    
                    # 删除临时图片目录
                    shutil.rmtree(picture_dir)
                    print(f">>> 已清理临时图片目录")
                else:
                    print(f"!!! ffmpeg 执行失败:\n{result.stderr}")
            except FileNotFoundError:
                print("!!! 未找到 ffmpeg，请先安装: sudo apt install ffmpeg")
            except Exception as e:
                print(f"!!! 视频生成失败: {e}")
        else:
            # 随机渲染指定数量的图片
            n_render = min(args.render_n, len(test_set.images))
            render_indices = random.sample(range(len(test_set.images)), n_render)
            
            print(f"\n>>> 渲染测试集图片（随机 {n_render} 张）...")
            os.makedirs(render_dir, exist_ok=True)
            psnrs = []
            
            model.eval()
            with torch.no_grad():
                for i, idx in enumerate(tqdm(render_indices)):
                    rays_o, rays_d, target = test_set.get_image_rays(idx, device)
                    H, W = rays_o.shape[:2]
                    rays_o = rays_o.reshape(-1, 3)
                    rays_d = rays_d.reshape(-1, 3)
                    
                    # 使用 density_grid 加速渲染
                    pred_chunks = []
                    for j in range(0, rays_o.shape[0], chunk):
                        pred_chunk, _, _ = render_rays(
                            model=model,
                            rays_o=rays_o[j:j+chunk],
                            rays_d=rays_d[j:j+chunk],
                            near=near,
                            far=far,
                            n_samples=render_n_samples,
                            perturb=False,
                            white_bkgd=white_bkgd,
                            density_grid=density_grid,  
                        )
                        pred_chunks.append(pred_chunk)
                    
                    pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                    pred = torch.clamp(pred, 0.0, 1.0)
                    psnr = compute_psnr_torch(pred, target)
                    psnrs.append(psnr)
                    
                    # 保存渲染图片（带PSNR信息）
                    plt.imsave(
                        os.path.join(render_dir, f"render_{idx:03d}_psnr{psnr:.2f}.png"),
                        pred.cpu().numpy(),
                    )
            
            avg_psnr = float(np.mean(psnrs))
            print(f"\n>>> 渲染完成！平均 PSNR: {avg_psnr:.2f} dB")
            print(f">>> 保存路径: {render_dir}")
        return
    
    # 训练后的标准评估：计算测试集PSNR
    model.eval()
    print(f"\n>>> 评估 {test_split} 集...")
    psnrs = []
    with torch.no_grad():
        for idx in tqdm(range(len(test_set))):
            rays_o, rays_d, target = test_set.get_image_rays(idx, device)
            H, W = rays_o.shape[:2]
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # 使用 density_grid 加速渲染
            pred_chunks = []
            for j in range(0, rays_o.shape[0], chunk):
                pred_chunk, _, _ = render_rays(
                    model=model,
                    rays_o=rays_o[j:j+chunk],
                    rays_d=rays_d[j:j+chunk],
                    near=near,
                    far=far,
                    n_samples=render_n_samples,
                    perturb=False,
                    white_bkgd=white_bkgd,
                    density_grid=density_grid,
                )
                pred_chunks.append(pred_chunk)
            
            pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
            pred = torch.clamp(pred, 0.0, 1.0)
            psnr = compute_psnr_torch(pred, target)
            psnrs.append(psnr)

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    print(f"\n{'='*60}")
    print(f">>> Instant-NeRF 评估结果")
    print(f">>> 测试集平均 PSNR: {avg_psnr:.2f} dB")
    print(f">>> 最佳验证集 PSNR: {best_val_psnr:.2f} dB" if not args.eval_only else "")
    print(f"{'='*60}")


def run_part3(cfg, args):
    """Part 3: 动态 NeRF"""
    if not args.data_dir:
        raise ValueError("Part 3 requires --data_dir pointing to a dynamic NeRF dataset root.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 读取渲染和训练配置
    downscale = cfg.get("downscale", 1)
    white_bkgd = cfg.get("white_bkgd", True)
    scene_scale = cfg.get("scene_scale", 1.0)
    near = float(cfg.get("near", 2.0))
    far = float(cfg.get("far", 6.0))
    n_samples = cfg.get("n_samples", 64)
    render_n_samples = cfg.get("render_n_samples", n_samples)
    batch_size = cfg.get("batch_size", 4096)
    train_iters = cfg.get("train_iters", 20000)
    learning_rate = cfg.get("learning_rate", 5e-4)
    log_every = cfg.get("log_every", 100)
    save_every = cfg.get("save_every", 2000)
    chunk = cfg.get("chunk", 8192)
    deformation_reg_weight = cfg.get("deformation_reg_weight", 1e-4) # 变形正则化权重
    render_n = args.render_n
    if args.render_chunk is not None:
        chunk = args.render_chunk
    log_dir = cfg.get("log_dir", "output/part3")
    
    # 获取数据集名称并添加到输出路径
    dataset_name = os.path.basename(args.data_dir)
    log_dir = os.path.join(log_dir, dataset_name)

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir)
    render_dir = os.path.join(log_dir, "renders")
    val_render_dir = os.path.join(log_dir, "val_renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(val_render_dir, exist_ok=True)

    from src.dataset import DynamicDataset
    
    train_set = DynamicDataset(
        root_dir=args.data_dir,
        split="train",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    # 加载验证集
    val_set = DynamicDataset(
        root_dir=args.data_dir,
        split="val",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    # 加载测试集
    test_split = "test"
    test_meta = os.path.join(args.data_dir, "transforms_test.json")
    if not os.path.exists(test_meta):
        test_split = "val"
    test_set = DynamicDataset(
        root_dir=args.data_dir,
        split=test_split,
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    if not args.eval_only:
        print(f">>> 数据集: 训练集 {len(train_set.images)} 张 | 验证集 {len(val_set.images)} 张 | 测试集 {len(test_set.images)} 张")
    else:
        print(f">>> 数据集: 测试集 {len(test_set.images)} 张")

    # 模型初始化
    from src.core import NeuralField
    model = NeuralField(cfg).to(device)
    
    # 如果使用 instant 模式，启用 density_grid
    canonical_type = cfg.get('canonical_type', 'nerf')
    density_grid = None
    active_ratio = 1.0
    if canonical_type == 'instant':
        use_density_grid = cfg.get('use_density_grid', True)
        if use_density_grid:
            from src.renderer import DensityGrid
            grid_resolution = cfg.get('grid_resolution', 128)
            grid_threshold = cfg.get('grid_threshold', 0.01)
            scene_bound = cfg.get('scene_bound', 1.5)
            density_grid = DensityGrid(
                resolution=grid_resolution,
                bound=scene_bound,
                threshold=grid_threshold
            ).to(device)
            print(f">>> Density Grid 已启用: {grid_resolution}³ 分辨率 (Instant-NGP 模式)")
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if density_grid is not None and "density_grid" in ckpt:
            density_grid.load_state_dict(ckpt["density_grid"])
        print(f">>> Loaded checkpoint: {args.checkpoint}")

    # 训练阶段
    if not args.eval_only:
        # 初始化 TensorBoard
        tb_dir = os.path.join(log_dir, "tensorboard", get_exp_name(cfg))
        tb_logger = TensorBoardLogger(tb_dir)
        
        weight_decay = cfg.get('weight_decay', 1e-5)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # TiNeuVox 改进：使用 CosineAnnealingLR 调度器，防止训练后期 PSNR 震荡
        # 从初始学习率平滑降至 eta_min
        eta_min = cfg.get('eta_min', 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=eta_min)
        
        use_amp = cfg.get('use_amp', True)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        loss_fn = nn.MSELoss()
        print(">>> 开始训练 Part 3 (Dynamic NeRF)...")
        print(f">>> tensorboard --logdir={os.path.join(log_dir, 'tensorboard')} 查看 TensorBoard 日志")
        
        # ======== 正则化和数据增强配置 ========
        
        # A. TV Loss (Total Variation) - 惩罚 HashGrid 相邻特征之间的差异，消除边缘毛刺和浮点噪声
        use_tv_loss = cfg.get('use_tv_loss', True) and canonical_type == 'instant'
        tv_loss_weight = cfg.get('tv_loss_weight', 1e-6)
        
        # B. 时间平滑正则化 - 确保运动在时间轴上二阶导数微小
        use_temporal_smooth = cfg.get('use_temporal_smooth', True)
        temporal_smooth_weight = cfg.get('temporal_smooth_weight', 1e-4)
        temporal_epsilon = cfg.get('temporal_epsilon', 0.02)  # 时间差 ε
        temporal_n_samples = cfg.get('temporal_n_samples', 256)  # 采样点数
        
        # C. 随机背景增强，每 batch 随机一个颜色
        use_random_bg = cfg.get('use_random_bg', False)
        random_bg_start = cfg.get('random_bg_start', 0) if use_random_bg else float('inf')
        
        # D. 无监督一致性约束（体积守恒）
        use_unsup_consistency = cfg.get('use_unsupervised_consistency', False)
        unsup_consistency_weight = cfg.get('unsup_consistency_weight', 0.001)
        unsup_n_samples = cfg.get('unsup_n_samples', 512)
        
        # 打印配置信息
        if use_tv_loss:
            print(f">>> 正则化: TV Loss 已启用 (weight={tv_loss_weight:.0e}, 消除空间噪声)")
        if use_temporal_smooth:
            print(f">>> 正则化: 时间平滑已启用 (weight={temporal_smooth_weight:.0e}, ε={temporal_epsilon}, 消除时间抖动)")
        if use_random_bg:
            if random_bg_start > 0:
                print(f">>> 数据增强: 随机背景增强 ({random_bg_start} 步后开启)")
            else:
                print(f">>> 数据增强: 随机背景增强 (全程启动)")
        if cfg.get('use_coord_noise', False):
            print(f">>> 数据增强: 坐标噪声已启用 (coord_std={cfg.get('coord_noise_std', 0.005)}, time_std={cfg.get('time_noise_std', 0.02)})")
        if use_unsup_consistency:
            print(f">>> 数据增强: 无监督一致性约束已启用 (weight={unsup_consistency_weight}, n_samples={unsup_n_samples})")
        
        # 初始化最佳验证集PSNR跟踪
        best_val_psnr = 0.0

        model.train()
        grid_update_interval = cfg.get('grid_update_interval', 16)
        grid_warmup_iters = cfg.get('grid_warmup_iters', 256)
        
        for step in range(1, train_iters + 1):
            # 采样返回: rays_o, rays_d, target_rgba [B,4], times
            rays_o, rays_d, target_rgba, times = train_set.sample_random_rays(batch_size, device)
            
            # 分离 RGB 和 Alpha 通道
            target_rgb = target_rgba[:, :3]    # [B, 3]
            target_alpha = target_rgba[:, 3:4] # [B, 1]
            
            # ======== B. 随机背景增强（学界标准做法）========
            # 从 random_bg_start 步开始启用随机背景增强
            if use_random_bg and step >= random_bg_start:
                bg_color = torch.rand(3, device=device)  # [3] 随机 RGB
            else:
                bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
            
            # 合成 target: Target = RGB * Alpha + bg_color * (1 - Alpha)
            target = target_rgb * target_alpha + bg_color * (1.0 - target_alpha)
            
            # 性能优化：使用混合精度前向传播
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 调用 render_rays，传入相同的 bg_color
                pred_rgb, _, _, extras = render_rays(
                    model=model,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    near=near,
                    far=far,
                    n_samples=n_samples,
                    perturb=True,
                    times=times,
                    density_grid=density_grid,
                    bg_color=bg_color,  # 传入随机背景色
                )
                
                # A. 辅助损失函数: RGB Loss + Deformation Regularization
                loss_rgb = loss_fn(pred_rgb, target)
                mean_delta_x = extras['mean_delta_x'] # 从 extras 获取加权平均变形量
                loss_reg = torch.mean(mean_delta_x ** 2) * deformation_reg_weight
                
                # TV Loss (Total Variation) - 惩罚 HashGrid 哈希表中相邻条目的特征差异
                loss_tv = torch.tensor(0.0, device=device)
                if use_tv_loss and hasattr(model, 'canonical_repr') and hasattr(model.canonical_repr, 'encoding'):
                    # 获取 HashGrid 的可学习参数
                    hash_params = model.canonical_repr.encoding.params  # [N_entries, n_features]
                    
                    # 计算相邻哈希条目之间的 L1 差异 (TV 范数) 并惩罚
                    tv_diff = torch.abs(hash_params[1:] - hash_params[:-1])  # [N-1, n_features]
                    loss_tv = torch.mean(tv_diff) * tv_loss_weight
                
                # 时间平滑正则化 - 要求运动在时间轴上是二阶导数微小的
                loss_temporal = torch.tensor(0.0, device=device)
                # 每 2 步计算一次，减少计算开销
                if use_temporal_smooth and step > grid_warmup_iters and step % 2 == 0:
                    n_temp = temporal_n_samples
                    scene_bound = cfg.get('scene_bound', 1.2)
                    
                    # 随机采样空间点（在场景边界内）
                    x_temp = (torch.rand(n_temp, 3, device=device) * 2 - 1) * scene_bound
                    
                    # 随机采样时间点 t，确保 t+ε 仍在 [0, 1] 范围内
                    t_temp = torch.rand(n_temp, 1, device=device) * (1.0 - temporal_epsilon)
                    t_temp_eps = t_temp + temporal_epsilon
                    
                    # 计算同一点在两个相邻时刻的位移
                    feat_x_temp = model.pos_encoder_for_deform(x_temp)
                    feat_t_temp = model.time_encoder(t_temp)
                    feat_t_temp_eps = model.time_encoder(t_temp_eps)
                    
                    delta_x_t = model.deform_net(feat_x_temp, feat_t_temp)        # D(x, t)
                    delta_x_t_eps = model.deform_net(feat_x_temp, feat_t_temp_eps)  # D(x, t+ε)
                    
                    # 使用 L2 范数惩罚差异
                    loss_temporal = torch.mean((delta_x_t - delta_x_t_eps) ** 2) * temporal_smooth_weight * 2  # *2 补偿采样频率
                
                # 无监督一致性约束（体积守恒）
                # 对随机时刻的变形场施加约束，要求位移均值趋近于 0。因为全局体积应该保持守恒，物体不应该凭空膨胀或收缩
                loss_unsup = torch.tensor(0.0, device=device)
                # 每 4 步计算一次，减少计算开销
                if use_unsup_consistency and step > grid_warmup_iters and step % 4 == 0:
                    n_unsup = min(unsup_n_samples, 512)
                    t_rand = torch.rand(n_unsup, 1, device=device)
                    scene_bound = cfg.get('scene_bound', 1.2)
                    x_rand = (torch.rand(n_unsup, 3, device=device) * 2 - 1) * scene_bound
                    
                    # 仅获取变形场的位移（不需要渲染）
                    feat_t_rand = model.time_encoder(t_rand)
                    feat_x_rand = model.pos_encoder_for_deform(x_rand)
                    delta_x_rand = model.deform_net(feat_x_rand, feat_t_rand)
                    
                    # 约束：变形量的全局均值应趋近于 0（体积守恒）
                    loss_unsup = torch.mean(torch.abs(delta_x_rand.mean(dim=0))) * unsup_consistency_weight * 4  # *4 补偿采样频率
                
                total_loss = loss_rgb + loss_reg + loss_tv + loss_temporal + loss_unsup

            optimizer.zero_grad()
            # 性能优化：使用混合精度反向传播
            scaler.scale(total_loss).backward()
            
            # 梯度裁剪防止 DeformNet 和 HashGrid 在动态场景中溢出
            max_grad_norm = cfg.get('max_grad_norm', 1.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 执行学习率调度
            
            # 分离loss值，防止保留计算图
            loss_rgb_val = loss_rgb.item()
            loss_reg_val = loss_reg.item()
            loss_tv_val = loss_tv.item() if use_tv_loss else 0.0
            loss_temporal_val = loss_temporal.item() if use_temporal_smooth else 0.0
            loss_unsup_val = loss_unsup.item() if use_unsup_consistency else 0.0
            total_loss_val = total_loss.item()
            
            # 只删除extras避免累积，其他变量让Python自动管理
            del extras
            
            # 性能优化：动态调整网格更新频率
            # 前 10% 步数：每 16 步更新（快速建立包络线）
            # 10%-50% 步数：每 64 步更新（中期优化）
            # 50% 步数后：每 256 步更新（后期微调）
            if step < train_iters * 0.1:
                dynamic_interval = 16
            elif step < train_iters * 0.5:
                dynamic_interval = 64
            else:
                dynamic_interval = 256
            
            if density_grid is not None and density_grid.should_update(step, dynamic_interval, grid_warmup_iters):
                model.eval()
                # TiNeuVox 改进：暴力时空更新 - 一次性采样多个时间点，让网格形成完整的"运动包络线"
                time_min = train_set.times.min().item()
                time_max = train_set.times.max().item()
                # 根据训练阶段动态调整采样密度
                n_time_samples = 16 if step < 1000 else 8
                update_times = torch.linspace(time_min, time_max, steps=n_time_samples, device=device)
                
                for i, t_val in enumerate(update_times):
                    # 完全禁用衰减：严格时空并集，永久保留所有时刻的密度
                    active_ratio = density_grid.update(
                        model, 
                        device=device, 
                        time=t_val.view(1, 1), 
                        decay=1.0  # 完全保留
                    )
                
                model.train()

            if step % log_every == 0:
                psnr = compute_psnr(loss_rgb_val)
                current_lr = scheduler.get_last_lr()[0]
                skip_info = ""
                if density_grid is not None:
                    skip_info = f" | Skip: {(1-active_ratio)*100:.1f}%"
                
                print(
                    f">>> Step {step}/{train_iters} | "
                    f"Loss {total_loss_val:.6f} | "
                    f"PSNR {psnr:.2f} dB | "
                    f"LR {current_lr:.6f}{skip_info}"
                )
                
                # 记录到 TensorBoard
                tb_logger.log_scalar('Train/RGB_Loss', loss_rgb_val, step)
                tb_logger.log_scalar('Train/Reg_Loss', loss_reg_val, step)
                tb_logger.log_scalar('Train/Total_Loss', total_loss_val, step)
                tb_logger.log_scalar('Train/PSNR', psnr, step)
                tb_logger.log_scalar('Train/LearningRate', current_lr, step)
                if use_tv_loss:
                    tb_logger.log_scalar('Train/TV_Loss', loss_tv_val, step)
                if use_temporal_smooth:
                    tb_logger.log_scalar('Train/Temporal_Loss', loss_temporal_val, step)
                if use_unsup_consistency:
                    tb_logger.log_scalar('Train/Unsup_Loss', loss_unsup_val, step)
                if density_grid is not None:
                    tb_logger.log_scalar('Train/ActiveRatio', active_ratio, step)
            
            # 定期验证集评估
            val_every = cfg.get("val_every", 500)
            if step % val_every == 0:
                model.eval()
                val_psnrs = []
                val_results = []  # 保存 (idx, psnr, pred_img, time) 用于后续保存
                
                # 对全部验证集计算 PSNR，随机保存 5 张图片
                import random
                n_save_images = min(5, len(val_set.images))
                save_indices = set(random.sample(range(len(val_set.images)), n_save_images))
                
                step_val_dir = os.path.join(val_render_dir, f"step_{step:06d}")
                os.makedirs(step_val_dir, exist_ok=True)
                
                with torch.no_grad():
                    # 验证时使用固定白色背景（保证公平对比）
                    val_bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
                    
                    # 对全部验证集计算 PSNR
                    for idx in range(len(val_set.images)):
                        rays_o, rays_d, target, time = val_set.get_image_rays(idx, device)
                        H, W = rays_o.shape[:2]
                        rays_o = rays_o.reshape(-1, 3)
                        rays_d = rays_d.reshape(-1, 3)
                        target_flat = target.reshape(-1, 3)
                        time = time.expand(H*W, 1)
                        
                        # 分块渲染验证集
                        pred_chunks = []
                        for i in range(0, rays_o.shape[0], chunk):
                            pred_chunk, _, _, _ = render_rays(
                                model=model,
                                rays_o=rays_o[i:i+chunk],
                                rays_d=rays_d[i:i+chunk],
                                near=near,
                                far=far,
                                n_samples=render_n_samples,
                                perturb=False,
                                times=time[i:i+chunk],
                                density_grid=density_grid,
                                bg_color=val_bg_color,  # 固定背景色
                            )
                            pred_chunks.append(pred_chunk.cpu())  # 立即移动到 CPU
                        pred = torch.cat(pred_chunks, dim=0)
                        del pred_chunks  # 立即释放
                        
                        val_psnr = compute_psnr_torch(pred.to(device), target_flat)
                        val_psnrs.append(val_psnr)
                        
                        # 只保存随机选中的图片
                        if idx in save_indices:
                            pred_img = pred.reshape(H, W, 3)
                            pred_img = torch.clamp(pred_img, 0.0, 1.0)
                            plt.imsave(
                                os.path.join(step_val_dir, f"val_{idx:03d}_t{time[0,0].item():.2f}_psnr{val_psnr:.2f}.png"),
                                pred_img.numpy(),
                            )
                        del pred, target_flat, rays_o, rays_d  # 清理显存
                
                avg_val_psnr = float(np.mean(val_psnrs))
                print(f"    [Validation] PSNR: {avg_val_psnr:.2f} dB", end="")
                
                # 记录验证集 PSNR 到 TensorBoard
                tb_logger.log_scalar('Validation/PSNR', avg_val_psnr, step)
                
                plt.close('all')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 只在验证集PSNR提升时保存模型
                if avg_val_psnr > best_val_psnr:
                    best_val_psnr = avg_val_psnr
                    best_path = os.path.join(ckpt_dir, f"best_model.pth")
                    save_dict = {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "val_psnr": best_val_psnr
                    }
                    if density_grid is not None:
                        save_dict["density_grid"] = density_grid.state_dict()
                    torch.save(save_dict, best_path)
                    print(f" | 🌟 New Best Model! Saved to {best_path}")
                else:
                    print()
                
                model.train()

        print(f"\n>>> 训练完成！最佳验证集 PSNR: {best_val_psnr:.2f} dB")
        tb_logger.close()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 评估阶段
    import shutil
    import subprocess
    import json
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.interpolate import interp1d
    
    # 清理训练集和验证集以节省显存（只保留测试集用于评估）
    del train_set, val_set
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    
    # 评估时使用固定背景色
    eval_bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
    
    # 创建临时图片目录用于生成视频
    picture_dir = os.path.join(log_dir, "picture")
    os.makedirs(picture_dir, exist_ok=True)
    
    # render_n == -1 时：环绕渲染视频
    if render_n == -1:
        # 从配置文件读取视频参数
        n_interp_frames = cfg.get('video_frames', 300)  # 视频总帧数
        n_rotations = cfg.get('n_rotations', 2)  # 旋转圈数
        print(f">>> 环绕渲染模式: 生成 {n_interp_frames} 帧，相机绕物体旋转 {n_rotations} 圈，时间 0→1...")
        
        # 从配置文件读取相机参数
        radius = cfg.get('camera_radius', 2.4)  # 相机环绕半径
        
        # 场景中心和相机高度
        scene_center = cfg.get('scene_center', [0.0, 0.0, 0.0])
        camera_height = cfg.get('camera_height', 2.8)
        center = np.array(scene_center)
        
        print(f">>> 环绕半径: {radius:.3f}")
        print(f">>> 场景中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f">>> 相机高度: {camera_height:.2f}")
        
        # 时间从 0 线性增长到 1
        interp_times = np.linspace(0.0, 1.0, n_interp_frames)
        
        # 相机绕 Z 轴旋转 n_rotations 圈 (0 到 n_rotations × 2π)
        angles = np.linspace(0.0, n_rotations * 2 * np.pi, n_interp_frames, endpoint=False)
        
        # 生成环绕相机位姿
        interp_poses = np.zeros((n_interp_frames, 4, 4), dtype=np.float32)
        for i, angle in enumerate(angles):
            # 相机位置：在 XY 平面上绕场景中心旋转
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = camera_height  # 保持恒定高度
            cam_pos = np.array([x, y, z])
            
            # 相机朝向场景中心（look-at）
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            # 世界坐标系的上方向
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # 构建旋转矩阵 (NeRF 相机坐标系: x=right, y=up, z=-forward)
            R = np.stack([right, up, -forward], axis=1)  # [3, 3]
            
            interp_poses[i, :3, :3] = R
            interp_poses[i, :3, 3] = cam_pos
            interp_poses[i, 3, 3] = 1.0
        
        # 渲染插值帧
        H, W = test_set.H, test_set.W
        focal = test_set.focal
        
        with torch.no_grad():
            for idx in tqdm(range(n_interp_frames), desc="Interpolated Rendering"):
                # 构建光线
                c2w = torch.tensor(interp_poses[idx], dtype=torch.float32, device=device)
                t = torch.tensor([[interp_times[idx]]], dtype=torch.float32, device=device)
                
                # 生成光线（与 dataset 中相同的逻辑）
                j, i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                dirs = torch.stack([
                    (i - W * 0.5) / focal,
                    -(j - H * 0.5) / focal,
                    -torch.ones_like(i),
                ], dim=-1).reshape(-1, 3)
                
                rays_d = torch.matmul(dirs, c2w[:3, :3].T)
                rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                rays_o = c2w[:3, 3].expand_as(rays_d)
                if test_set.scene_scale != 1.0:
                    rays_o = rays_o * test_set.scene_scale
                
                time_batch = t.expand(H*W, 1)
                
                # 分块渲染
                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    pred_chunk, _, _, _ = render_rays(
                        model=model,
                        rays_o=rays_o[i:i+chunk],
                        rays_d=rays_d[i:i+chunk],
                        near=near,
                        far=far,
                        n_samples=render_n_samples,
                        perturb=False,
                        times=time_batch[i:i+chunk],
                        density_grid=density_grid,
                        bg_color=eval_bg_color,
                    )
                    pred_chunks.append(pred_chunk)
                
                pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                pred = torch.clamp(pred, 0.0, 1.0)
                
                plt.imsave(
                    os.path.join(picture_dir, f"frame_{idx:03d}.png"),
                    pred.cpu().numpy(),
                )
                
                # 清理显存防止泄漏
                del pred, pred_chunks, rays_o, rays_d, time_batch, c2w, t, dirs
                torch.cuda.empty_cache()
        
        print(f">>> 插值渲染完成！共 {n_interp_frames} 帧")
        psnrs = []  # 插值模式没有 ground truth，无法计算 PSNR
    else:
        # 正常模式：渲染指定数量的测试集帧
        print(f">>> Rendering {test_split} set...")
        psnrs = []
        num_renders = min(render_n, len(test_set))
        
        with torch.no_grad():
            for idx in tqdm(range(num_renders), desc="Rendering"):
                rays_o, rays_d, target, time = test_set.get_image_rays(idx, device)
                H, W = rays_o.shape[:2]
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                time = time.expand(H*W, 1)

                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    pred_chunk, _, _, _ = render_rays(
                        model=model,
                        rays_o=rays_o[i:i+chunk],
                        rays_d=rays_d[i:i+chunk],
                        near=near,
                        far=far,
                        n_samples=render_n_samples,
                        perturb=False,
                        times=time[i:i+chunk],
                        density_grid=density_grid,
                        bg_color=eval_bg_color,
                    )
                    pred_chunks.append(pred_chunk)
                
                pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                pred = torch.clamp(pred, 0.0, 1.0)
                psnr = compute_psnr_torch(pred, target)
                psnrs.append(psnr)
                
                # 保存为连续编号的帧（用于生成视频）
                plt.imsave(
                    os.path.join(picture_dir, f"frame_{idx:03d}.png"),
                    pred.cpu().numpy(),
                )
                # 同时保存带时间戳的版本
                plt.imsave(
                    os.path.join(render_dir, f"{test_split}_{idx:03d}_t{time[0,0].item():.2f}.png"),
                    pred.cpu().numpy(),
                )
                
                del pred, pred_chunks, rays_o, rays_d, target, time
                torch.cuda.empty_cache()
        
        num_frames = num_renders

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    if psnrs:
        print(f"\n>>> Test PSNR: {avg_psnr:.2f} dB")
    print(f">>> Rendered images saved to: {picture_dir}")
    
    # 使用 ffmpeg 生成视频
    dataset_name = os.path.basename(args.data_dir)
    video_path = os.path.join(log_dir, f"{dataset_name}_24fps.mp4")
    print(f"\n>>> 使用 ffmpeg 生成视频...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "24",
            "-i", os.path.join(picture_dir, "frame_%03d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f">>> 视频已保存: {video_path}")
            print(f">>> 视频时长: {n_interp_frames/24:.1f}秒 ({n_interp_frames} 帧 @ 24fps)")
            
            # 删除临时图片目录
            shutil.rmtree(picture_dir)
            print(f">>> 已清理临时图片目录")
        else:
            print(f"!!! ffmpeg 执行失败:\n{result.stderr}")
    except FileNotFoundError:
        print("!!! 未找到 ffmpeg，请先安装: sudo apt install ffmpeg")
    except Exception as e:
        print(f"!!! 视频生成失败: {e}")


def run_part4(cfg, args):
    """
    Part 4: Dual-Hash Dynamic NeRF (创新点：哈希位移场 + 哈希规范场)
    
    核心创新：
    1. Dual-Hash 协同架构：用 HashGrid 替代 MLP 变形网络
    2. TV-Displacement Loss：对位移网格施加全变分正则化
    3. 时空解耦设计：空间位移由 HashGrid 查询，时间调制由轻量 MLP 完成
    """
    if not args.data_dir:
        raise ValueError("Part 4 requires --data_dir pointing to a dynamic NeRF dataset root.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")
    print(">>> Part 4: Dual-Hash Dynamic NeRF")

    # 读取渲染和训练配置
    downscale = cfg.get("downscale", 1)
    white_bkgd = cfg.get("white_bkgd", True)
    scene_scale = cfg.get("scene_scale", 1.0)
    near = float(cfg.get("near", 2.0))
    far = float(cfg.get("far", 6.0))
    n_samples = cfg.get("n_samples", 64)
    render_n_samples = cfg.get("render_n_samples", n_samples)
    batch_size = cfg.get("batch_size", 4096)
    train_iters = cfg.get("train_iters", 20000)
    learning_rate = cfg.get("learning_rate", 5e-4)
    log_every = cfg.get("log_every", 100)
    chunk = cfg.get("chunk", 8192)
    render_n = args.render_n
    if args.render_chunk is not None:
        chunk = args.render_chunk
    log_dir = cfg.get("log_dir", "output/part4")
    
    # 获取数据集名称并添加到输出路径
    dataset_name = os.path.basename(args.data_dir)
    log_dir = os.path.join(log_dir, dataset_name)

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir)
    render_dir = os.path.join(log_dir, "renders")
    val_render_dir = os.path.join(log_dir, "val_renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(val_render_dir, exist_ok=True)

    from src.dataset import DynamicDataset
    
    train_set = DynamicDataset(
        root_dir=args.data_dir,
        split="train",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    val_set = DynamicDataset(
        root_dir=args.data_dir,
        split="val",
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    test_split = "test"
    test_meta = os.path.join(args.data_dir, "transforms_test.json")
    if not os.path.exists(test_meta):
        test_split = "val"
    test_set = DynamicDataset(
        root_dir=args.data_dir,
        split=test_split,
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )
    
    if not args.eval_only:
        print(f">>> 数据集: 训练集 {len(train_set.images)} 张 | 验证集 {len(val_set.images)} 张 | 测试集 {len(test_set.images)} 张")
    else:
        print(f">>> 数据集: 测试集 {len(test_set.images)} 张")

    # 模型初始化
    from src.core import NeuralField
    model = NeuralField(cfg).to(device)
    
    # 启用 density_grid（与 Part 3 Instant 相同）
    use_density_grid = cfg.get('use_density_grid', True)
    density_grid = None
    active_ratio = 1.0
    if use_density_grid:
        from src.renderer import DensityGrid
        grid_resolution = cfg.get('grid_resolution', 128)
        grid_threshold = cfg.get('grid_threshold', 0.01)
        scene_bound = cfg.get('scene_bound', 1.5)
        density_grid = DensityGrid(
            resolution=grid_resolution,
            bound=scene_bound,
            threshold=grid_threshold
        ).to(device)
        print(f">>> Density Grid 已启用: {grid_resolution}³ 分辨率")
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if density_grid is not None and "density_grid" in ckpt:
            density_grid.load_state_dict(ckpt["density_grid"])
            # 更新 active_ratio 以反映加载的 density_grid 状态
            active_ratio = density_grid.binary_grid.float().mean().item()
        print(f">>> Loaded checkpoint: {args.checkpoint}")

    # =========================================================================
    # 训练阶段
    # =========================================================================
    if not args.eval_only:
        tb_dir = os.path.join(log_dir, "tensorboard", get_exp_name(cfg))
        tb_logger = TensorBoardLogger(tb_dir)
        
        weight_decay = cfg.get('weight_decay', 1e-5)
        
        # ==============================================================
        # Part 4 分组学习率优化（兼容单网格和三网格模式）
        # ==============================================================
        param_groups = []
        
        # 1. 三网格模式：分别设置学习率
        for grid_name in ['deform_grid_start', 'deform_grid_mid', 'deform_grid_end']:
            if hasattr(model, grid_name):
                grid = getattr(model, grid_name)
                param_groups.append({
                    'params': grid.parameters(),
                    'lr': learning_rate * 2.0,
                    'name': grid_name
                })
        
        # 2. 单网格模式兼容（如果没有三网格，使用 deformation_grid）
        if not hasattr(model, 'deform_grid_start') and hasattr(model, 'deformation_grid'):
            param_groups.append({
                'params': model.deformation_grid.parameters(),
                'lr': learning_rate * 2.0,
                'name': 'deformation_grid'
            })
        
        # 3. 规范空间哈希网格：高学习率
        if hasattr(model, 'canonical_repr'):
            param_groups.append({
                'params': model.canonical_repr.parameters(),
                'lr': learning_rate * 2.0,  # 2x 基础学习率
                'name': 'canonical_repr'
            })
        
        # 3. displacement_scale：超高学习率（标量参数学习慢）
        if hasattr(model, 'deform_decoder'):
            param_groups.append({
                'params': [model.deform_decoder.displacement_scale],
                'lr': learning_rate * 5.0,  # 5x 基础学习率
                'name': 'displacement_scale'
            })
            # deform_net 用正常学习率
            param_groups.append({
                'params': [p for n, p in model.deform_decoder.named_parameters() if 'displacement_scale' not in n],
                'lr': learning_rate,
                'name': 'deform_decoder'
            })
        
        # 4. 其他参数（时间调制、解码器等）：正常学习率
        excluded_names = {'deform_grid_start', 'deform_grid_mid', 'deform_grid_end', 
                         'deformation_grid', 'canonical_repr', 'deform_decoder'}
        other_params = [p for n, p in model.named_parameters() 
                       if not any(ex in n for ex in excluded_names)]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': learning_rate,
                'name': 'others'
            })
        
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        
        eta_min = cfg.get('eta_min', 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=eta_min)
        
        use_amp = cfg.get('use_amp', True)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        loss_fn = nn.MSELoss()
        
        # ==============================================================================
        # 正则化配置
        # ==============================================================================
        
        # 1. TV-Displacement Loss（核心创新）
        # 对位移哈希网格施加全变分正则化，强制相邻格点位移一致
        use_tv_displacement = cfg.get('use_tv_displacement', True)
        tv_displacement_weight = cfg.get('tv_displacement_weight', 0.001)
        
        # 2. 规范空间 TV Loss
        tv_loss_weight = cfg.get('tv_loss_weight', 1e-5)
        
        # 3. 变形场 L2 正则化
        deformation_reg_weight = cfg.get('deformation_reg_weight', 0.01)
        
        # 4. 时间平滑正则化
        use_temporal_smooth = cfg.get('use_temporal_smooth', True)
        temporal_smooth_weight = cfg.get('temporal_smooth_weight', 1e-4)
        temporal_epsilon = cfg.get('temporal_epsilon', 0.02)
        temporal_n_samples = cfg.get('temporal_n_samples', 256)
        
        # 5. 随机背景增强
        use_random_bg = cfg.get('use_random_bg', False)
        random_bg_start = cfg.get('random_bg_start', 0) if use_random_bg else float('inf')
        
        # 6. 无监督一致性约束
        use_unsup_consistency = cfg.get('use_unsupervised_consistency', False)
        unsup_consistency_weight = cfg.get('unsup_consistency_weight', 0.001)
        unsup_n_samples = cfg.get('unsup_n_samples', 512)
        
        # 7. 静态锚点损失（强制 t=0 时零位移）
        use_static_anchor = cfg.get('use_static_anchor', True)
        static_anchor_weight = cfg.get('static_anchor_weight', 0.01)
        static_anchor_n_samples = cfg.get('static_anchor_n_samples', 512)
        
        # 打印配置
        print(">>> 开始训练 Part 4 (Dual-Hash Dynamic NeRF)...")
        print(f">>> tensorboard --logdir={os.path.join(log_dir, 'tensorboard')} 查看日志")
        if use_tv_displacement:
            print(f">>> 正则化: TV-Displacement Loss 已启用 (weight={tv_displacement_weight:.0e})")
        if tv_loss_weight > 0:
            print(f">>> 正则化: 规范空间 TV Loss (weight={tv_loss_weight:.0e})")
        if use_temporal_smooth:
            print(f">>> 正则化: 时间平滑 (weight={temporal_smooth_weight:.0e}, ε={temporal_epsilon})")
        if use_static_anchor:
            print(f">>> 正则化: 静态锚点损失已启用 (weight={static_anchor_weight:.0e}, t=0 时强制零位移)")
        if use_random_bg:
            print(f">>> 数据增强: 随机背景 ({random_bg_start} 步后开启)")
        if cfg.get('use_coord_noise', False):
            print(f">>> 数据增强: 坐标噪声 (coord={cfg.get('coord_noise_std', 0.005)}, time={cfg.get('time_noise_std', 0.02)})")
        
        best_val_psnr = 0.0
        model.train()
        grid_update_interval = cfg.get('grid_update_interval', 32)
        grid_warmup_iters = cfg.get('grid_warmup_iters', 256)
        
        for step in range(1, train_iters + 1):
            rays_o, rays_d, target_rgba, times = train_set.sample_random_rays(batch_size, device)
            
            target_rgb = target_rgba[:, :3]
            target_alpha = target_rgba[:, 3:4]
            
            # 随机背景
            if use_random_bg and step >= random_bg_start:
                bg_color = torch.rand(3, device=device)
            else:
                bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
            
            target = target_rgb * target_alpha + bg_color * (1.0 - target_alpha)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_rgb, _, _, extras = render_rays(
                    model=model,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    near=near,
                    far=far,
                    n_samples=n_samples,
                    perturb=True,
                    times=times,
                    density_grid=density_grid,
                    bg_color=bg_color,
                )
                
                # A. RGB Loss
                loss_rgb = loss_fn(pred_rgb, target)
                
                # B. 变形场 L2 正则化
                mean_delta_x = extras['mean_delta_x']
                loss_reg = torch.mean(mean_delta_x ** 2) * deformation_reg_weight
                

                #  TV-Displacement Loss
                # 对位移哈希网格的参数施加全变分正则化，强制相邻哈希条目的位移向量相似，消除边缘闪烁
                loss_tv_disp = torch.tensor(0.0, device=device)
                if use_tv_displacement:
                    # 三网格 TV Loss：对三个锚点网格分别施加 TV 正则化
                    tv_total = 0.0
                    for grid_name in ['deform_grid_start', 'deform_grid_mid', 'deform_grid_end']:
                        if hasattr(model, grid_name):
                            grid = getattr(model, grid_name)
                            params = grid.encoding.params
                            tv_diff = torch.abs(params[1:] - params[:-1])
                            tv_total = tv_total + torch.mean(tv_diff)
                    loss_tv_disp = tv_total * tv_displacement_weight / 3.0  # 平均
                
                # D. 规范空间 TV Loss
                loss_tv_canon = torch.tensor(0.0, device=device)
                if tv_loss_weight > 0 and hasattr(model, 'canonical_repr'):
                    canon_params = model.canonical_repr.encoding.params
                    tv_diff_canon = torch.abs(canon_params[1:] - canon_params[:-1])
                    loss_tv_canon = torch.mean(tv_diff_canon) * tv_loss_weight
                
                # E. 时间平滑正则化（每 16 步计算一次，大幅减少开销）
                loss_temporal = torch.tensor(0.0, device=device)
                if use_temporal_smooth and step > grid_warmup_iters and step % 16 == 0:
                    n_temp = 64  # 减少采样点数
                    scene_bound = cfg.get('scene_bound', 1.5)
                    
                    x_temp = (torch.rand(n_temp, 3, device=device) * 2 - 1) * scene_bound
                    t_temp = torch.rand(n_temp, 1, device=device) * (1.0 - temporal_epsilon)
                    t_temp_eps = t_temp + temporal_epsilon
                    
                    # Part 4 使用哈希位移场
                    feat_t = model.time_encoder(t_temp)
                    feat_t_eps = model.time_encoder(t_temp_eps)
                    time_mod = model.time_modulation(feat_t)
                    time_mod_eps = model.time_modulation(feat_t_eps)
                    
                    deform_feat = model.deformation_grid(x_temp)
                    delta_x_t = model.deform_decoder(deform_feat, time_mod)
                    delta_x_t_eps = model.deform_decoder(deform_feat, time_mod_eps)
                    
                    loss_temporal = torch.mean((delta_x_t - delta_x_t_eps) ** 2) * temporal_smooth_weight * 16  # 补偿采样频率
                
                # F. 无监督一致性约束（每 32 步计算一次）
                loss_unsup = torch.tensor(0.0, device=device)
                if use_unsup_consistency and step > grid_warmup_iters and step % 32 == 0:
                    n_unsup = 128  # 减少采样点数
                    t_rand = torch.rand(n_unsup, 1, device=device)
                    scene_bound = cfg.get('scene_bound', 1.5)
                    x_rand = (torch.rand(n_unsup, 3, device=device) * 2 - 1) * scene_bound
                    
                    feat_t_rand = model.time_encoder(t_rand)
                    time_mod_rand = model.time_modulation(feat_t_rand)
                    deform_feat_rand = model.deformation_grid(x_rand)
                    delta_x_rand = model.deform_decoder(deform_feat_rand, time_mod_rand)
                    
                    loss_unsup = torch.mean(torch.abs(delta_x_rand.mean(dim=0))) * unsup_consistency_weight * 32  # 补偿采样频率
                
                # ==============================================================
                # ⭐ 三网格锚点约束 (Tri-Grid Anchor Loss)
                # 对三个网格在各自的锚点时刻施加约束：
                #   - Grid_start: t=0 时强制零位移（定义规范空间）
                #   - Grid_mid:   t=1/2 时位移应平滑连续
                #   - Grid_end:   t=1 时无特殊约束（非循环场景）
                # ==============================================================
                loss_anchor = torch.tensor(0.0, device=device)
                if use_static_anchor and step > grid_warmup_iters and step % 16 == 0:
                    n_anchor = 128
                    scene_bound = cfg.get('scene_bound', 1.5)
                    
                    # 随机采样空间点
                    x_anchor = (torch.rand(n_anchor, 3, device=device) * 2 - 1) * scene_bound
                    
                    # ======== 1. t=0 强制零位移（核心约束）========
                    # t=0 落在 [0, 1/6] 段，100% 使用 Grid_start
                    t_zero = torch.zeros(n_anchor, 1, device=device)
                    feat_t_zero = model.time_encoder(t_zero)
                    time_mod_zero = model.time_modulation(feat_t_zero)
                    deform_feat_start = model.deform_grid_start(x_anchor)
                    delta_x_zero = model.deform_decoder(deform_feat_start, time_mod_zero)
                    loss_start = torch.mean(delta_x_zero ** 2)
                    
                    # ======== 2. 三网格一致性约束（可选）========
                    # 让三个网格在 t=1/6 时刻输出相近的特征，确保插值过渡平滑
                    # 这是软约束，权重较小
                    t_anchor = torch.full((n_anchor, 1), 1.0/6.0, device=device)
                    feat_t_anchor = model.time_encoder(t_anchor)
                    time_mod_anchor = model.time_modulation(feat_t_anchor)
                    
                    # 在 t=1/6 时，start 和 mid 网格应该有相似的"趋势"
                    delta_start_anchor = model.deform_decoder(model.deform_grid_start(x_anchor), time_mod_anchor)
                    delta_mid_anchor = model.deform_decoder(model.deform_grid_mid(x_anchor), time_mod_anchor)
                    # 软约束：两个网格在边界时刻的输出差异不要太大
                    loss_consistency = torch.mean((delta_start_anchor - delta_mid_anchor) ** 2) * 0.1
                    
                    # 总锚点损失
                    loss_anchor = (loss_start + loss_consistency) * static_anchor_weight * 16
                
                total_loss = loss_rgb + loss_reg + loss_tv_disp + loss_tv_canon + loss_temporal + loss_unsup + loss_anchor

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            max_grad_norm = cfg.get('max_grad_norm', 1.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # 分离 loss 值
            loss_rgb_val = loss_rgb.item()
            loss_reg_val = loss_reg.item()
            loss_tv_disp_val = loss_tv_disp.item() if use_tv_displacement else 0.0
            loss_tv_canon_val = loss_tv_canon.item() if tv_loss_weight > 0 else 0.0
            loss_temporal_val = loss_temporal.item() if use_temporal_smooth else 0.0
            loss_unsup_val = loss_unsup.item() if use_unsup_consistency else 0.0
            loss_anchor_val = loss_anchor.item() if use_static_anchor else 0.0
            total_loss_val = total_loss.item()
            
            del extras
            
            # 动态网格更新
            if step < train_iters * 0.1:
                dynamic_interval = 16
            elif step < train_iters * 0.5:
                dynamic_interval = 64
            else:
                dynamic_interval = 256
            
            grid_stop_ratio = cfg.get('grid_stop_ratio', 0.9)
            if density_grid is not None and step < train_iters * grid_stop_ratio and density_grid.should_update(step, dynamic_interval, grid_warmup_iters):
                model.eval()
                # ⭐ 三网格架构：只需采样三个锚点时刻即可覆盖运动轨迹
                # 避免过度采样导致 Skip 率暴跌
                anchor_times = torch.tensor([1.0/6.0, 0.5, 5.0/6.0], device=device)
                
                # 每 500 步启用自动剪枝，避免 Skip 率暴跌
                enable_prune = (step % 500 == 0) and (step > grid_warmup_iters)
                
                for t_val in anchor_times:
                    active_ratio = density_grid.update(
                        model, device=device, time=t_val.view(1, 1), decay=1.0,
                        auto_prune=enable_prune, threshold_multiplier=1.0
                    )
                model.train()

            if step % log_every == 0:
                psnr = compute_psnr(loss_rgb_val)
                current_lr = scheduler.get_last_lr()[0]
                skip_info = f" | Skip: {(1-active_ratio)*100:.1f}%" if density_grid else ""
                
                print(
                    f">>> Step {step}/{train_iters} | "
                    f"Loss {total_loss_val:.6f} | "
                    f"PSNR {psnr:.2f} dB | "
                    f"LR {current_lr:.6f}{skip_info}"
                )
                
                tb_logger.log_scalar('Train/RGB_Loss', loss_rgb_val, step)
                tb_logger.log_scalar('Train/Reg_Loss', loss_reg_val, step)
                tb_logger.log_scalar('Train/Total_Loss', total_loss_val, step)
                tb_logger.log_scalar('Train/PSNR', psnr, step)
                tb_logger.log_scalar('Train/LearningRate', current_lr, step)
                if use_tv_displacement:
                    tb_logger.log_scalar('Train/TV_Displacement_Loss', loss_tv_disp_val, step)
                if tv_loss_weight > 0:
                    tb_logger.log_scalar('Train/TV_Canon_Loss', loss_tv_canon_val, step)
                if use_temporal_smooth:
                    tb_logger.log_scalar('Train/Temporal_Loss', loss_temporal_val, step)
                if use_unsup_consistency:
                    tb_logger.log_scalar('Train/Unsup_Loss', loss_unsup_val, step)
                if use_static_anchor:
                    tb_logger.log_scalar('Train/Anchor_Loss', loss_anchor_val, step)
                if density_grid is not None:
                    tb_logger.log_scalar('Train/ActiveRatio', active_ratio, step)
            
            # 验证集评估
            val_every = cfg.get("val_every", 500)
            if step % val_every == 0:
                model.eval()
                val_psnrs = []
                
                import random
                n_save_images = min(5, len(val_set.images))
                save_indices = set(random.sample(range(len(val_set.images)), n_save_images))
                
                step_val_dir = os.path.join(val_render_dir, f"step_{step:06d}")
                os.makedirs(step_val_dir, exist_ok=True)
                
                with torch.no_grad():
                    val_bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
                    
                    for idx in range(len(val_set.images)):
                        rays_o, rays_d, target, time = val_set.get_image_rays(idx, device)
                        H, W = rays_o.shape[:2]
                        rays_o = rays_o.reshape(-1, 3)
                        rays_d = rays_d.reshape(-1, 3)
                        target_flat = target.reshape(-1, 3)
                        time = time.expand(H*W, 1)
                        
                        pred_chunks = []
                        for i in range(0, rays_o.shape[0], chunk):
                            pred_chunk, _, _, _ = render_rays(
                                model=model,
                                rays_o=rays_o[i:i+chunk],
                                rays_d=rays_d[i:i+chunk],
                                near=near,
                                far=far,
                                n_samples=render_n_samples,
                                perturb=False,
                                times=time[i:i+chunk],
                                density_grid=density_grid,
                                bg_color=val_bg_color,
                            )
                            pred_chunks.append(pred_chunk.cpu())
                        pred = torch.cat(pred_chunks, dim=0)
                        del pred_chunks
                        
                        val_psnr = compute_psnr_torch(pred.to(device), target_flat)
                        val_psnrs.append(val_psnr)
                        
                        if idx in save_indices:
                            pred_img = pred.reshape(H, W, 3)
                            pred_img = torch.clamp(pred_img, 0.0, 1.0)
                            plt.imsave(
                                os.path.join(step_val_dir, f"val_{idx:03d}_t{time[0,0].item():.2f}_psnr{val_psnr:.2f}.png"),
                                pred_img.numpy(),
                            )
                        del pred, target_flat, rays_o, rays_d
                
                avg_val_psnr = float(np.mean(val_psnrs))
                print(f"    [Validation] PSNR: {avg_val_psnr:.2f} dB", end="")
                
                tb_logger.log_scalar('Validation/PSNR', avg_val_psnr, step)
                
                plt.close('all')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if avg_val_psnr > best_val_psnr:
                    best_val_psnr = avg_val_psnr
                    best_path = os.path.join(ckpt_dir, f"best_model.pth")
                    save_dict = {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "val_psnr": best_val_psnr
                    }
                    if density_grid is not None:
                        save_dict["density_grid"] = density_grid.state_dict()
                    torch.save(save_dict, best_path)
                    print(f" | 🌟 New Best! Saved to {best_path}")
                else:
                    print()
                
                model.train()

        print(f"\n>>> 训练完成！最佳验证集 PSNR: {best_val_psnr:.2f} dB")
        tb_logger.close()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 评估阶段
    # =========================================================================
    import shutil
    import subprocess
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.interpolate import interp1d
    
    del train_set, val_set
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    eval_bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
    
    # eval_only 模式：只测试 PSNR，不生成视频
    if args.eval_only:
        print(f"\n>>> 评估模式：计算测试集 PSNR...")
        psnrs = []
        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), desc="Evaluating"):
                rays_o, rays_d, target, time = test_set.get_image_rays(idx, device)
                H, W = rays_o.shape[:2]
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                time = time.expand(H*W, 1)

                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    pred_chunk, _, _, _ = render_rays(
                        model=model,
                        rays_o=rays_o[i:i+chunk],
                        rays_d=rays_d[i:i+chunk],
                        near=near,
                        far=far,
                        n_samples=render_n_samples,
                        perturb=False,
                        times=time[i:i+chunk],
                        density_grid=density_grid,
                        bg_color=eval_bg_color,
                    )
                    pred_chunks.append(pred_chunk)
                
                pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                pred = torch.clamp(pred, 0.0, 1.0)
                psnr = compute_psnr_torch(pred, target)
                psnrs.append(psnr)
                
                del pred, pred_chunks, rays_o, rays_d, target, time
                torch.cuda.empty_cache()
        
        avg_psnr = float(np.mean(psnrs))
        print(f"\n{'='*60}")
        print(f">>> Part 4 测试集评估结果")
        print(f">>> 平均 PSNR: {avg_psnr:.2f} dB ({len(psnrs)} 张图片)")
        print(f"{'='*60}")
        return
    
    # 训练模式结束后直接返回，不生成视频
    if not args.eval_only:
        print(f"\n>>> 训练完成！使用 --eval_only --render_n -1 来生成视频")
        return
    
    # --eval_only + render_n != -1：渲染指定数量的测试集图片
    # --eval_only + render_n == -1：生成环绕视频
    picture_dir = os.path.join(log_dir, "picture")
    os.makedirs(picture_dir, exist_ok=True)
    
    if render_n == -1:
        n_interp_frames = cfg.get('video_frames', 300)
        n_rotations = cfg.get('n_rotations', 2)
        print(f">>> 环绕渲染模式: 生成 {n_interp_frames} 帧，相机绕物体旋转 {n_rotations} 圈...")
        
        radius = cfg.get('camera_radius', 2.4)
        scene_center = cfg.get('scene_center', [0.0, 0.0, 0.0])
        camera_height = cfg.get('camera_height', 2.8)
        center = np.array(scene_center)
        
        print(f">>> 环绕半径: {radius:.3f}, 场景中心: {center}, 相机高度: {camera_height:.2f}")
        
        interp_times = np.linspace(0.0, 1.0, n_interp_frames)
        angles = np.linspace(0.0, n_rotations * 2 * np.pi, n_interp_frames, endpoint=False)
        
        interp_poses = np.zeros((n_interp_frames, 4, 4), dtype=np.float32)
        for i, angle in enumerate(angles):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = camera_height
            cam_pos = np.array([x, y, z])
            
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            R = np.stack([right, up, -forward], axis=1)
            interp_poses[i, :3, :3] = R
            interp_poses[i, :3, 3] = cam_pos
            interp_poses[i, 3, 3] = 1.0
        
        H, W = test_set.H, test_set.W
        focal = test_set.focal
        
        with torch.no_grad():
            for idx in tqdm(range(n_interp_frames), desc="Rendering"):
                c2w = torch.tensor(interp_poses[idx], dtype=torch.float32, device=device)
                t = torch.tensor([[interp_times[idx]]], dtype=torch.float32, device=device)
                
                j, i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                dirs = torch.stack([
                    (i - W * 0.5) / focal,
                    -(j - H * 0.5) / focal,
                    -torch.ones_like(i),
                ], dim=-1).reshape(-1, 3)
                
                rays_d = torch.matmul(dirs, c2w[:3, :3].T)
                rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                rays_o = c2w[:3, 3].expand_as(rays_d)
                if test_set.scene_scale != 1.0:
                    rays_o = rays_o * test_set.scene_scale
                
                time_batch = t.expand(H*W, 1)
                
                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    pred_chunk, _, _, _ = render_rays(
                        model=model,
                        rays_o=rays_o[i:i+chunk],
                        rays_d=rays_d[i:i+chunk],
                        near=near,
                        far=far,
                        n_samples=render_n_samples,
                        perturb=False,
                        times=time_batch[i:i+chunk],
                        density_grid=density_grid,
                        bg_color=eval_bg_color,
                    )
                    pred_chunks.append(pred_chunk)
                
                pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                pred = torch.clamp(pred, 0.0, 1.0)
                
                plt.imsave(
                    os.path.join(picture_dir, f"frame_{idx:03d}.png"),
                    pred.cpu().numpy(),
                )
                
                del pred, pred_chunks, rays_o, rays_d, time_batch, c2w, t, dirs
                torch.cuda.empty_cache()
        
        print(f">>> 渲染完成！共 {n_interp_frames} 帧")
        psnrs = []
    else:
        print(f">>> Rendering {test_split} set...")
        psnrs = []
        num_renders = min(render_n, len(test_set))
        
        with torch.no_grad():
            for idx in tqdm(range(num_renders), desc="Rendering"):
                rays_o, rays_d, target, time = test_set.get_image_rays(idx, device)
                H, W = rays_o.shape[:2]
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                time = time.expand(H*W, 1)

                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    pred_chunk, _, _, _ = render_rays(
                        model=model,
                        rays_o=rays_o[i:i+chunk],
                        rays_d=rays_d[i:i+chunk],
                        near=near,
                        far=far,
                        n_samples=render_n_samples,
                        perturb=False,
                        times=time[i:i+chunk],
                        density_grid=density_grid,
                        bg_color=eval_bg_color,
                    )
                    pred_chunks.append(pred_chunk)
                
                pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
                pred = torch.clamp(pred, 0.0, 1.0)
                psnr = compute_psnr_torch(pred, target)
                psnrs.append(psnr)
                
                plt.imsave(os.path.join(picture_dir, f"frame_{idx:03d}.png"), pred.cpu().numpy())
                plt.imsave(os.path.join(render_dir, f"{test_split}_{idx:03d}_t{time[0,0].item():.2f}.png"), pred.cpu().numpy())
                
                del pred, pred_chunks, rays_o, rays_d, target, time
                torch.cuda.empty_cache()
        
        n_interp_frames = num_renders

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    if psnrs:
        print(f"\n>>> Test PSNR: {avg_psnr:.2f} dB")
    print(f">>> Rendered images saved to: {picture_dir}")
    
    # 生成视频
    dataset_name = os.path.basename(args.data_dir)
    video_path = os.path.join(log_dir, f"{dataset_name}_part4_24fps.mp4")
    print(f"\n>>> 使用 ffmpeg 生成视频...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "24",
            "-i", os.path.join(picture_dir, "frame_%03d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f">>> 视频已保存: {video_path}")
            print(f">>> 视频时长: {n_interp_frames/24:.1f}秒 ({n_interp_frames} 帧 @ 24fps)")
            shutil.rmtree(picture_dir)
            print(f">>> 已清理临时图片目录")
        else:
            print(f"!!! ffmpeg 执行失败:\n{result.stderr}")
    except FileNotFoundError:
        print("!!! 未找到 ffmpeg，请先安装: sudo apt install ffmpeg")
    except Exception as e:
        print(f"!!! 视频生成失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="输入图像路径 (Part 1)")
    parser.add_argument("--data_dir", type=str, help="NeRF 数据集根目录 (Part 2)")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="加载已训练模型")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅评估/渲染，不进行训练（需 --checkpoint）",
    )
    parser.add_argument("--render_n", type=int, default=-1, help="评估时渲染的测试集图片数量，如果为 -1 则插值渲染 300 帧") 
    parser.add_argument("--render_chunk", type=int, help="覆盖渲染 chunk 大小")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("mode")
    if mode == "part1_fourier":
        if not args.image:
            raise ValueError("Part 1 requires --image.")
        if args.eval_only and not args.checkpoint:
            raise ValueError("Part 1 eval_only requires --checkpoint.")
        run_part1(cfg, args)
    elif mode == "part2_nerf":
        if args.eval_only and not args.checkpoint:
            raise ValueError("eval_only requires --checkpoint.")
        run_part2(cfg, args)
    elif mode == "part2_instant":
        if args.eval_only and not args.checkpoint:
            raise ValueError("eval_only requires --checkpoint.")
        run_part2_instant(cfg, args)
    elif mode == "part3":
        if args.eval_only and not args.checkpoint:
            raise ValueError("eval_only requires --checkpoint.")
        run_part3(cfg, args)
    elif mode == "part4":
        if args.eval_only and not args.checkpoint:
            raise ValueError("eval_only requires --checkpoint.")
        run_part4(cfg, args)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
