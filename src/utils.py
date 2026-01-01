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
