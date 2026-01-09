"""工具函数"""
import torch
import numpy as np

def compute_psnr(mse):
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    PSNR = 10 * log10(MAX^2 / MSE)
    对于归一化到 [0,1] 的图像，MAX = 1
    
    Args:
        mse: Mean Squared Error
    Returns:
        PSNR value in dB
    """
    return 10 * np.log10(1.0 / mse)

def compute_psnr_torch(pred, target):
    """
    直接从预测值和目标值计算 PSNR
    
    Args:
        pred: 预测图像 [H, W, 3] 或 [N, 3]
        target: 目标图像 [H, W, 3] 或 [N, 3]
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2).item()
    return compute_psnr(mse)


def render_image_safe(
    render_fn,
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    chunk,
    white_bkgd,
):
    """
    带自动OOM恢复的图像渲染
    
    当显存不足时自动减半chunk并重试，避免程序崩溃
    """
    chunk_size = int(chunk)
    while True:
        try:
            return render_fn(
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
            # 显存不足时自动减半chunk并重试
            if not torch.cuda.is_available():
                raise
            if chunk_size <= 1024:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(chunk_size // 2, 1024)
            print(f">>> CUDA OOM, reducing render chunk to {chunk_size}")


