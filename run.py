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
from src.utils import compute_psnr, compute_psnr_torch


def get_2d_data(path, max_size=400):
    img = Image.open(path).convert("RGB")
    w_orig, h_orig = img.size
    scale = min(max_size / w_orig, max_size / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    img_np = np.array(img) / 255.0
    h, w, _ = img_np.shape
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)
    rgb = torch.tensor(img_np.reshape(-1, 3), dtype=torch.float32)
    return coords, rgb, h, w


def run_part1(cfg, args):
    epochs = cfg["epochs"]
    learning_rate = cfg["learning_rate"]
    batch_size = cfg.get("batch_size", None)
    image_size = cfg.get("image_size", 400)
    log_dir = cfg.get("log_dir", "output/")
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

    coords, gt_rgb, h, w = get_2d_data(args.image, max_size=image_size)
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
            run_dir = os.path.join(log_dir, "part1", run_name)
            os.makedirs(run_dir, exist_ok=True)

            save_intermediate = isinstance(save_every, int) and save_every > 0
            if save_intermediate:
                steps_dir = os.path.join(run_dir, "steps")
                os.makedirs(steps_dir, exist_ok=True)

            print(f">>> [{run_idx}/{len(param_combos)}] 配置: {run_name}, Steps={epochs}")

            model = NeuralField(config).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for i in tqdm(range(epochs)):
                if batch_size is None:
                    pred_rgb = model(coords)
                    loss = loss_fn(pred_rgb, gt_rgb)
                else:
                    indices = torch.randint(0, total_pixels, (batch_size,), device=device)
                    batch_coords = coords[indices]
                    batch_gt_rgb = gt_rgb[indices]
                    pred_rgb = model(batch_coords)
                    loss = loss_fn(pred_rgb, batch_gt_rgb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if save_intermediate and (i + 1) % save_every == 0:
                    with torch.no_grad():
                        intermediate_img = model(coords).cpu().numpy().reshape(h, w, 3)
                    plt.imsave(
                        os.path.join(steps_dir, f"step_{i+1:05d}.png"),
                        intermediate_img,
                    )

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


def render_image_safe(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    chunk,
    white_bkgd,
):
    chunk_size = int(chunk)
    while True:
        try:
            return render_image(
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
                chunk=chunk_size,
                white_bkgd=white_bkgd,
            )
        except torch.cuda.OutOfMemoryError:
            if not torch.cuda.is_available():
                raise
            if chunk_size <= 1024:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(chunk_size // 2, 1024)
            print(f">>> CUDA OOM, reducing render chunk to {chunk_size}")


def run_part2(cfg, args):
    if not args.data_dir:
        raise ValueError("Part 2 requires --data_dir pointing to a NeRF dataset root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

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
    log_dir = cfg.get("log_dir", "output/part2")
    if args.render_chunk:
        chunk = args.render_chunk

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    render_dir = os.path.join(log_dir, "renders")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

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

    model = NeuralField(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f">>> Loaded checkpoint: {args.checkpoint}")

    if not args.eval_only:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        print(">>> Start Training Part 2 (NeRF)...")
        model.train()
        for step in range(1, train_iters + 1):
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

    model.eval()
    print(f">>> Rendering {test_split} set...")
    psnrs = []
    with torch.no_grad():
        for idx in range(len(test_set)):
            rays_o, rays_d, target = test_set.get_image_rays(idx, device)
            pred = render_image_safe(
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


def main():
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
