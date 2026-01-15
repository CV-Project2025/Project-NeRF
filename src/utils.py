"""工具函数"""
import torch
import numpy as np
import os
import warnings

# 消除 TensorBoard 警告信息
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def get_exp_name(cfg):
    """获取实验名称（从配置或自动生成时间戳）"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.get("exp_name", timestamp)


class TensorBoardLogger:
    """简化的 TensorBoard 日志记录器"""
    def __init__(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("!!! TensorBoard 未安装，日志功能已禁用")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """记录标量值"""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """记录多个标量值"""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """关闭 writer"""
        if self.enabled and self.writer is not None:
            self.writer.close()
