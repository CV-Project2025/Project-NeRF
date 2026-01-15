import json
import os

import numpy as np
import torch
from PIL import Image


class BlenderDataset:
    """
    NeRF Synthetic/Blender 数据集加载器
    
    加载多视角图像和相机位姿，生成用于训练的光线
    """
    def __init__(
        self,
        root_dir,
        split="train",
        downscale=1,
        white_bkgd=True,
        scene_scale=1.0,
    ):
        self.root_dir = root_dir
        self.split = split
        self.downscale = max(int(downscale), 1)
        self.white_bkgd = white_bkgd
        self.scene_scale = float(scene_scale)

        # 读取相机参数和图像元数据
        meta_path = os.path.join(root_dir, f"transforms_{split}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.camera_angle_x = float(meta["camera_angle_x"])
        self.frames = meta["frames"]

        # 加载所有图像和位姿
        images = []
        poses = []
        for frame in self.frames:
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]

            # 自动检测文件扩展名
            img_path = os.path.join(root_dir, file_path)
            if not os.path.splitext(img_path)[1]:
                if os.path.exists(img_path + ".png"):
                    img_path += ".png"
                elif os.path.exists(img_path + ".jpg"):
                    img_path += ".jpg"

            # 加载并降采样图像
            img = Image.open(img_path).convert("RGBA")
            if self.downscale > 1:
                img = img.resize(
                    (img.width // self.downscale, img.height // self.downscale),
                    Image.LANCZOS,
                )

            # 处理透明通道：alpha 合成到背景
            img = np.array(img).astype(np.float32) / 255.0
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            if self.white_bkgd:
                rgb = rgb * alpha + (1.0 - alpha)  # 白色背景
            else:
                rgb = rgb * alpha  # 黑色背景

            images.append(torch.from_numpy(rgb))
            poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

        self.images = torch.stack(images, dim=0)
        self.poses = torch.stack(poses, dim=0)
        self.H, self.W = self.images.shape[1:3]
        
        # 根据视场角计算焦距
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)

        # 预计算所有像素的相机空间方向
        self._directions = self._build_directions()

    def _build_directions(self):
        """
        构建相机空间的光线方向 (针孔相机模型)
        
        坐标系: x右, y下, z指向屏幕内
        """
        j, i = torch.meshgrid(
            torch.arange(self.H), torch.arange(self.W), indexing="ij"
        )
        # 将像素坐标转换为归一化相机坐标
        dirs = torch.stack(
            [
                (i - self.W * 0.5) / self.focal,  # x: 水平方向
                -(j - self.H * 0.5) / self.focal,  # y: 垂直方向 (上为负)
                -torch.ones_like(i),  # z: 指向场景 (负 z 轴)
            ],
            dim=-1,
        )
        return dirs

    def __len__(self):
        return self.images.shape[0]

    def get_rays(self, c2w):
        """
        根据相机位姿生成光线
        
        Args:
            c2w: [4, 4] camera-to-world 变换矩阵
        Returns:
            rays_o: [H, W, 3] 光线起点 (相机位置)
            rays_d: [H, W, 3] 光线方向 (已归一化)
        """
        # 将 directions 移动到与 c2w 相同的设备
        directions = self._directions.to(c2w.device).reshape(-1, 3)
        # 将相机空间方向转换到世界空间
        rays_d = torch.matmul(directions, c2w[:3, :3].T)
        rays_d = rays_d.reshape(self.H, self.W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # 光线起点即相机位置
        rays_o = c2w[:3, 3].expand_as(rays_d)
        if self.scene_scale != 1.0:
            rays_o = rays_o * self.scene_scale
        return rays_o, rays_d

    def get_image_rays(self, index, device):
        """获取指定图像的所有光线及目标颜色"""
        c2w = self.poses[index]
        rays_o, rays_d = self.get_rays(c2w)
        target = self.images[index]
        return rays_o.to(device), rays_d.to(device), target.to(device)

    def sample_random_rays(self, batch_size, device):
        """
        随机采样光线用于训练
        
        从所有图像中随机选择像素，返回对应的光线和目标颜色
        """
        # 随机选择图像和像素位置
        img_idx = torch.randint(0, len(self), (batch_size,))
        pix_y = torch.randint(0, self.H, (batch_size,))
        pix_x = torch.randint(0, self.W, (batch_size,))

        # 计算选中像素的光线方向
        c2w = self.poses[img_idx]
        dirs = torch.stack(
            [
                (pix_x - self.W * 0.5) / self.focal,
                -(pix_y - self.H * 0.5) / self.focal,
                -torch.ones_like(pix_x),
            ],
            dim=-1,
        )
        # 批量矩阵乘法: [B, 3, 3] × [B, 3, 1] → [B, 3, 1]
        rays_d = torch.bmm(c2w[:, :3, :3], dirs.unsqueeze(-1)).squeeze(-1)
        rays_o = c2w[:, :3, 3]
        if self.scene_scale != 1.0:
            rays_o = rays_o * self.scene_scale

        # 获取对应像素的目标颜色
        target = self.images[img_idx, pix_y, pix_x]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        return rays_o.to(device), rays_d.to(device), target.to(device)


class DynamicDataset(BlenderDataset):
    """动态场景数据集，继承自 BlenderDataset 并增加时间戳支持"""
    def __init__(self, root_dir, split="train", downscale=1, white_bkgd=True, scene_scale=1.0):
        super().__init__(root_dir, split, downscale, white_bkgd, scene_scale)
        
        # 尝试从元数据中提取时间戳
        times = []
        for frame in self.frames:
            # 如果元数据中有 'time' 字段，则直接使用
            if 'time' in frame:
                times.append(frame['time'])
            else:
                # 否则，根据帧索引归一化生成时间戳 [0, 1]
                # 这假设帧是按时间顺序排列的
                times.append(len(times) / len(self.frames))
        
        self.times = torch.tensor(times, dtype=torch.float32)

    def get_image_rays(self, index, device):
        """获取指定图像的所有光线、目标颜色和对应的时间戳"""
        rays_o, rays_d = self.get_rays(self.poses[index])
        target = self.images[index]
        time = self.times[index].view(1, 1)  # 形状 [1, 1] 便于后续广播
        return rays_o.to(device), rays_d.to(device), target.to(device), time.to(device)

    def sample_random_rays(self, batch_size, device):
        """随机采样光线用于训练，并返回对应的时间戳"""
        img_idx = torch.randint(0, len(self), (batch_size,))
        pix_y = torch.randint(0, self.H, (batch_size,))
        pix_x = torch.randint(0, self.W, (batch_size,))

        # 计算光线方向
        c2w = self.poses[img_idx]
        dirs = torch.stack([
            (pix_x - self.W * 0.5) / self.focal,
            -(pix_y - self.H * 0.5) / self.focal,
            -torch.ones_like(pix_x),
        ], dim=-1)
        rays_d = torch.bmm(c2w[:, :3, :3], dirs.unsqueeze(-1)).squeeze(-1)
        rays_o = c2w[:, :3, 3]
        if self.scene_scale != 1.0:
            rays_o = rays_o * self.scene_scale

        target = self.images[img_idx, pix_y, pix_x]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # 获取时间戳并调整形状为 [batch_size, 1]
        times = self.times[img_idx].unsqueeze(-1)

        return rays_o.to(device), rays_d.to(device), target.to(device), times.to(device)