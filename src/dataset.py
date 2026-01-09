import json
import os

import numpy as np
import torch
from PIL import Image


class BlenderDataset:
    """NeRF Synthetic/Blender dataset loader."""
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

        meta_path = os.path.join(root_dir, f"transforms_{split}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.camera_angle_x = float(meta["camera_angle_x"])
        self.frames = meta["frames"]

        images = []
        poses = []
        for frame in self.frames:
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]

            img_path = os.path.join(root_dir, file_path)
            if not os.path.splitext(img_path)[1]:
                if os.path.exists(img_path + ".png"):
                    img_path += ".png"
                elif os.path.exists(img_path + ".jpg"):
                    img_path += ".jpg"

            img = Image.open(img_path).convert("RGBA")
            if self.downscale > 1:
                img = img.resize(
                    (img.width // self.downscale, img.height // self.downscale),
                    Image.LANCZOS,
                )

            img = np.array(img).astype(np.float32) / 255.0
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            if self.white_bkgd:
                rgb = rgb * alpha + (1.0 - alpha)
            else:
                rgb = rgb * alpha

            images.append(torch.from_numpy(rgb))
            poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

        self.images = torch.stack(images, dim=0)
        self.poses = torch.stack(poses, dim=0)
        self.H, self.W = self.images.shape[1:3]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)

        self._directions = self._build_directions()

    def _build_directions(self):
        j, i = torch.meshgrid(
            torch.arange(self.H), torch.arange(self.W), indexing="ij"
        )
        dirs = torch.stack(
            [
                (i - self.W * 0.5) / self.focal,
                -(j - self.H * 0.5) / self.focal,
                -torch.ones_like(i),
            ],
            dim=-1,
        )
        return dirs

    def __len__(self):
        return self.images.shape[0]

    def get_rays(self, c2w):
        directions = self._directions.reshape(-1, 3)
        rays_d = torch.matmul(directions, c2w[:3, :3].T)
        rays_d = rays_d.reshape(self.H, self.W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = c2w[:3, 3].expand_as(rays_d)
        if self.scene_scale != 1.0:
            rays_o = rays_o * self.scene_scale
        return rays_o, rays_d

    def get_image_rays(self, index, device):
        c2w = self.poses[index]
        rays_o, rays_d = self.get_rays(c2w)
        target = self.images[index]
        return rays_o.to(device), rays_d.to(device), target.to(device)

    def sample_random_rays(self, batch_size, device):
        img_idx = torch.randint(0, len(self), (batch_size,))
        pix_y = torch.randint(0, self.H, (batch_size,))
        pix_x = torch.randint(0, self.W, (batch_size,))

        c2w = self.poses[img_idx]
        dirs = torch.stack(
            [
                (pix_x - self.W * 0.5) / self.focal,
                -(pix_y - self.H * 0.5) / self.focal,
                -torch.ones_like(pix_x),
            ],
            dim=-1,
        )
        rays_d = torch.bmm(c2w[:, :3, :3], dirs.unsqueeze(-1)).squeeze(-1)
        rays_o = c2w[:, :3, 3]
        if self.scene_scale != 1.0:
            rays_o = rays_o * self.scene_scale

        target = self.images[img_idx, pix_y, pix_x]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        return rays_o.to(device), rays_d.to(device), target.to(device)
