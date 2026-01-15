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
from src.utils import compute_psnr, compute_psnr_torch, render_image_safe, TensorBoardLogger


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

    total_pixels = coords.shape[0]
    print(">>> Start Training Part 1 (2D Fitting)...")
    print(
        f">>> 图像尺寸: {h}x{w}, 批量大小: {'全图' if batch_size is None else batch_size}"
    )
    print(f">>> 参数组合数: {len(param_combos)}")

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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        print(">>> Start Training Part 2 (NeRF)...")
        model.train()
        for step in range(1, train_iters + 1):
            # 随机采样光线并渲染
            rays_o, rays_d, target = train_set.sample_random_rays(batch_size, device)
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

            if save_every and step % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"model_step_{step:06d}.pth")
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path
                )

        final_path = os.path.join(ckpt_dir, "model_final.pth")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, final_path)

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
            print(f">>> Loaded checkpoint with DensityGrid: {args.checkpoint} (Step {ckpt.get("step", "未知")} | Val PSNR {ckpt.get("val_psnr", None):.2f} dB)")
        else:
            print(f">>> Loaded checkpoint: {args.checkpoint} (Step {ckpt.get("step", "未知")} | Val PSNR {ckpt.get("val_psnr", None):.2f} dB)")

    # 训练阶段
    if not args.eval_only:
        # 初始化 TensorBoard
        tb_dir = os.path.join(log_dir, "tensorboard")
        tb_logger = TensorBoardLogger(tb_dir)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        print(f">>> 目标: {train_iters} 步")
        print(f">>> 学习率: {learning_rate} ")
        print(f">>> 批量大小: {batch_size}")
        print(f">>> 采样点数: {n_samples} ")
        print(f">>>  tensorboard --logdir={tb_dir} 查看 TensorBoard 日志")
        
        # 初始化最佳验证集PSNR跟踪
        best_val_psnr = 0.0
        
        model.train()
        for step in range(1, train_iters + 1):
            # 随机采样光线并渲染
            rays_o, rays_d, target = train_set.sample_random_rays(batch_size, device)
            
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
            )
            loss = loss_fn(pred_rgb, target)

            optimizer.zero_grad()
            loss.backward()
            
            # 分别裁剪散列表和 MLP 的梯度
            if hasattr(model, 'representation'):
                torch.nn.utils.clip_grad_norm_(model.representation.parameters(), max_norm=1.0)
            if hasattr(model, 'decoder'):
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 定期更新 Density Grid（warmup 后才开始）
            if density_grid is not None and density_grid.should_update(step, grid_update_interval, grid_warmup_iters):
                model.eval()
                active_ratio = density_grid.update(model, device=device, time=None)
                model.train()

            # 日志输出和 TensorBoard 记录
            if step % log_every == 0:
                psnr = compute_psnr(loss.item())
                skip_info = ""
                if density_grid is not None:
                    skip_info = f" | Skip: {(1-active_ratio)*100:.1f}%"
                print(
                    f">>> Step {step}/{train_iters} | Loss {loss.item():.6f} | PSNR {psnr:.2f} dB{skip_info}"
                )
                
                # 记录到 TensorBoard
                tb_logger.log_scalar('Train/Loss', loss.item(), step)
                tb_logger.log_scalar('Train/PSNR', psnr, step)
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

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    from src.dataset import DynamicDataset
    
    train_set = DynamicDataset(
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
    test_set = DynamicDataset(
        root_dir=args.data_dir,
        split=test_split,
        downscale=downscale,
        white_bkgd=white_bkgd,
        scene_scale=scene_scale,
    )

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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        print(">>> 开始训练 Part 3 (Dynamic NeRF)...")

        model.train()
        grid_update_interval = cfg.get('grid_update_interval', 16)
        grid_warmup_iters = cfg.get('grid_warmup_iters', 256)
        
        for step in range(1, train_iters + 1):
            rays_o, rays_d, target, times = train_set.sample_random_rays(batch_size, device)
            
            # 调用修改后的 render_rays，接收 extras
            pred_rgb, _, _, extras = render_rays(
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
                perturb=True,
                white_bkgd=white_bkgd,
                times=times,
                density_grid=density_grid,
            )
            
            # A. 辅助损失函数: RGB Loss + Deformation Regularization
            loss_rgb = loss_fn(pred_rgb, target)
            mean_delta_x = extras['mean_delta_x'] # 从 extras 获取加权平均变形量
            loss_reg = torch.mean(mean_delta_x ** 2) * deformation_reg_weight
            total_loss = loss_rgb + loss_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if density_grid is not None and density_grid.should_update(step, grid_update_interval, grid_warmup_iters):
                model.eval()
                # 随机采样一个时刻进行增量更新,多次迭代后会自动形成运动轨迹的时空并集
                time_min = train_set.times.min().item()
                time_max = train_set.times.max().item()
                rand_time = torch.rand(1, 1, device=device) * (time_max - time_min) + time_min
                active_ratio = density_grid.update(model, device=device, time=rand_time, decay=0.95)
                model.train()

            if step % log_every == 0:
                psnr = compute_psnr(loss_rgb.item())
                skip_info = ""
                if density_grid is not None:
                    skip_info = f" | Skip: {(1-active_ratio)*100:.1f}%"
                print(
                    f">>> Step {step}/{train_iters} | "
                    f"RGB Loss {loss_rgb.item():.6f} | "
                    f"Reg Loss {loss_reg.item():.9f} | "
                    f"PSNR {psnr:.2f} dB{skip_info}"
                )

            if save_every and step % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"model_step_{step:06d}.pth")
                save_dict = {"model_state_dict": model.state_dict(), "config": cfg}
                if density_grid is not None:
                    save_dict["density_grid"] = density_grid.state_dict()
                torch.save(save_dict, ckpt_path)

        final_path = os.path.join(ckpt_dir, "model_final.pth")
        save_dict = {"model_state_dict": model.state_dict(), "config": cfg}
        if density_grid is not None:
            save_dict["density_grid"] = density_grid.state_dict()
        torch.save(save_dict, final_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 评估阶段
    model.eval()
    print(f">>> Rendering {test_split} set...")
    psnrs = []
    
    with torch.no_grad():
        num_renders = len(test_set) if render_n == -1 else min(render_n, len(test_set))
        for idx in range(num_renders):
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
                    white_bkgd=white_bkgd,
                    times=time[i:i+chunk],
                    density_grid=density_grid,
                )
                pred_chunks.append(pred_chunk)
            
            pred = torch.cat(pred_chunks, dim=0).reshape(H, W, 3)
            pred = torch.clamp(pred, 0.0, 1.0)
            psnr = compute_psnr_torch(pred, target)
            psnrs.append(psnr)
            
            plt.imsave(
                os.path.join(render_dir, f"{test_split}_{idx:03d}_t{time[0,0]:.2f}.png"),
                pred.cpu().numpy(),
            )

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    print(f">>> Test PSNR: {avg_psnr:.2f} dB")
    print(f">>> Rendered images saved to: {render_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="输入图像路径 (Part 1)")
    parser.add_argument("--data_dir", type=str, help="NeRF 数据集根目录 (Part 2)")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="加载 Part 2 已训练模型")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅渲染测试集，不进行训练（需 --checkpoint）",
    )
    parser.add_argument("--render_n", type=int, default=10, help="评估时渲染的测试集图片数量，如果为 -1 则渲染全部") 
    parser.add_argument("--render_chunk", type=int, help="覆盖渲染 chunk 大小")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("mode")
    if mode == "part1_fourier":
        if not args.image:
            raise ValueError("Part 1 requires --image.")
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
