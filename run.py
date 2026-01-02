import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import argparse
import yaml

from src.core import NeuralField
from src.utils import compute_psnr

# 简单的 DataLoader
def get_2d_data(path, max_size=400):
    img = Image.open(path).convert('RGB')
    W_orig, H_orig = img.size
    scale = min(max_size / W_orig, max_size / H_orig)
    new_W, new_H = int(W_orig * scale), int(H_orig * scale)
    img = img.resize((new_W, new_H), Image.LANCZOS)
    
    img_np = np.array(img) / 255.0
    H, W, _ = img_np.shape
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'
    ), dim=-1).reshape(-1, 2)
    rgb = torch.tensor(img_np.reshape(-1, 3), dtype=torch.float32)
    return coords, rgb, H, W

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    args = parser.parse_args()

    # 从 YAML 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    config = {
        'mode': cfg['mode'],
        'L_embed': cfg['L_embed'],
        'hidden_dim': cfg['hidden_dim'],
        'output_dim': cfg['output_dim'],
        'num_layers': cfg.get('num_layers', 3),
        'use_positional_encoding': cfg.get('use_positional_encoding', True)
    }
    
    epochs = cfg['epochs']
    learning_rate = cfg['learning_rate']
    batch_size = cfg.get('batch_size', None)
    image_size = cfg.get('image_size', 400)
    log_dir = cfg.get('log_dir', 'output/')
    save_every = cfg.get('save_every', 500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> 使用设备: {device}")
    
    # 初始化模型
    model = NeuralField(config).to(device)
    
    # 数据加载
    coords, gt_rgb, H, W = get_2d_data(args.image, max_size=image_size)
    coords, gt_rgb = coords.to(device), gt_rgb.to(device)
    
    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    steps_dir = os.path.join(log_dir, 'logs', 'steps')
    os.makedirs(steps_dir, exist_ok=True)

    # 训练
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    total_pixels = coords.shape[0]
    print(">>> Start Training Part 1 (2D Fitting)...")
    print(f">>> 配置: L={config['L_embed']}, Hidden={config['hidden_dim']}, Layers={config['num_layers']}, Steps={epochs}")
    print(f">>> 图像尺寸: {H}x{W}, 批量大小: {'全图' if batch_size is None else batch_size}")
    
    for i in tqdm(range(epochs)):
        if batch_size is None:
            # 全图训练
            pred_rgb = model(coords)
            loss = loss_fn(pred_rgb, gt_rgb)
        else:
            # 批量训练
            indices = torch.randint(0, total_pixels, (batch_size,), device=device)
            batch_coords = coords[indices]
            batch_gt_rgb = gt_rgb[indices]
            pred_rgb = model(batch_coords)
            loss = loss_fn(pred_rgb, batch_gt_rgb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 中间保存
        if (i + 1) % save_every == 0:
            with torch.no_grad():
                intermediate_img = model(coords).cpu().numpy().reshape(H, W, 3)
            plt.imsave(os.path.join(steps_dir, f'step_{i+1:05d}.png'), intermediate_img)

    # 最终保存
    with torch.no_grad():
        final_img = model(coords).cpu().numpy().reshape(H, W, 3)
    plt.imsave(os.path.join(log_dir, 'result_part1.png'), final_img)
    
    # 计算最终 PSNR
    final_psnr = compute_psnr(loss.item())
    print(f">>> Done! Final PSNR: {final_psnr:.2f} dB")

if __name__ == '__main__':
    main()