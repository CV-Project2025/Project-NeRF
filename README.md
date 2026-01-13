# Project-NeRF

---

## 1. 资源说明

#### 项目网盘

[这里](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 存放了部分数据集和预训练模型，结果。

#### 项目报告

[milestone](https://latex.pku.edu.cn/project/6954fa6e57c9c512c3b063aa)

---

## 2. 快速开始

### 2.1 环境配置

#### 基本配置

```bash
# 1. 克隆代码
git clone https://github.com/CV-Project2025/Project-NeRF
cd Project-NeRF

# 2. 创建并激活环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

#### Instant-NeRF 依赖安装

若要运行 **`Instant-NeRF`** ，请从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载 `install_ngp.sh` 放置到项目文件夹运行一下，以确保所需的 TinyCUDA-NN 和相关组件正确安装。

#### 配置自定义

将 `configs/part{num}.yaml.example` **复制**一份并重命名为 `configs/part{num}.yaml`，并根据需要修改配置文件中的参数。

### 2.2 Part 1: 2D 图像拟合

**1. 数据准备**
将目标图像（如 `fox.jpg`）放入 `data/` 目录。可以使用网盘提供的测试图。

**2. 训练**

```bash
# 执行训练并拟合
python3 run.py --image data/fox.jpg --config configs/part1.yaml
```

**3. 测试**
Part 1 训练过程中会自动进行拟合效果渲染。如果需要手动使用 checkpoint，可以运行：

```bash
python3 run.py --image data/fox.jpg --config configs/part1.yaml --checkpoint <path_to_model> --eval_only
```

**4. 输出说明**

- 训练日志与 TensorBoard 文件保存在 `output/runs/`。
- 每个 `save_every` 步数会保存当前拟合的图片到 `output/runs/<run_id>/steps/`。
- 最终结果保存为 `final.png`。
- PSNR 值越高（如 > 35dB）代表拟合精度越高。

### 2.3 Part 2: 神经辐射场 (NeRF)

**1. 数据准备**

建议下载 NeRF Synthetic 数据集，放置在 `data/nerf_synthetic/`下，可以从 [网盘/part2/data](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 获取。

**2. 训练**

```bash
# 标准 NeRF 训练
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml
```

**3. 测试**

```bash
# 使用已有模型进行测试
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml
  --checkpoint output/part2/checkpoints/model_step_020000.pth --eval_only
```

**4. 输出说明**

- 训练输出默认保存在 output/part2/ 目录
- 测试集渲染结果位于 output/part2/renders/
- 若数据集中没有 transforms_test.json，程序会自动使用 val 集合进行评估
- 若显存不足，可在运行时指定更小的渲染块大小，例如 --render_chunk 4096

### 2.4 Part 2: Instant-NeRF

**1. 数据准备**

数据同标准 NeRF，注意要保证相关 [依赖](#instant-nerf-依赖安装) 已安装。

**2. 训练**

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2_instant.yaml
```

**3. 测试**

```bash
# --render_n 参数：-1表示生成视频，正数表示随机渲染指定数量的视角
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2_instant.yaml
  --checkpoint output/part2_instant/checkpoints/model_best.pth --eval_only --render_n -1
```

**4. 输出说明**

- 结果保存在 `output/part2_instant/`。
- 使用 TensorBoard 监控损失下降与 PSNR 实时变化。
- 训练和渲染速度较快，可根据需要调整 [配置](configs/part2_instant.yaml) 中 `batch_size` 和 `chunk` 参数以适配显存。

---

## 3. 项目架构

```
Project-NeRF/
├── configs/                # 实验配置文件
│   ├── part1.yaml.example          # 2D 拟合配置模板
│   ├── part2.yaml.example          # 标准 NeRF 配置模板
│   └── part2_instant.yaml.example  # Instant-NGP 配置模板
├── data/                   # 数据存放目录
├── output/                 # 实验输出 (日志, checkpionts, 渲染图)
├── src/                    # 核心代码
│   ├── __init__.py
│   ├── abstract.py         # 抽象基类定义
│   ├── core.py             # 神经场核心逻辑 (组装embeddings, decoders)
│   ├── dataset.py          # 数据加载
│   ├── decoders.py         # 网络结构
│   ├── embeddings.py       # 编码器
│   ├── renderer.py         # 渲染器
│   └── utils.py            # 通用工具
├── run.py                  # 统一运行入口
├── requirements.txt
└── README.md
```

---

## 4. 实验原理与架构介绍

### Part 1: 2D 图像拟合 (Image Fitting)

**实验目标**：让神经网络学习并记住一张二维图像的像素分布 $(x, y) \to (r, g, b)$。

**核心原理**：

1.  **频谱偏差 (Spectral Bias)**：标准 MLP 倾向于学习低频信号，导致直接输入坐标无法拟合高频细节（图像模糊）。
2.  **位置编码 (Positional Encoding)**：通过傅里叶特征映射将低维坐标映射到高维频率空间，使网络能捕获高频细节。
    $$ \gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), ..., \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)) $$

**架构设计**：

- **输入**：像素坐标 $(x, y)$，归一化到 $[-1, 1]$。
- **编码层**：`FourierRepresentation`，将坐标扩展为 $2+4L$ 维向量。
- **网络**：`StandardMLP`，全连接层 + ReLU 激活。
- **输出**：RGB 颜色值。

### Part 2: 神经辐射场 (NeRF)

**实验目标**：通过稀疏的多视角 2D 图像，重建 3D 场景的几何与外观。

**核心原理**：

1.  **体渲染 (Volume Rendering)**：物理模型，通过沿着光线积分密度 ($\sigma$) 和颜色 ($c$) 来合成图像颜色。
    $$ C(r) = \int\_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt $$
2.  **光线投射 (Ray Marching)**：在每条光线上采样点，输入网络查询密度和颜色，最后累加。

**架构设计**：

- **输入**：空间坐标 $(x, y, z)$ 和 观察方向 $(\theta, \phi)$。
- **编码层**：对位置和方向分别进行位置编码。
- **网络**：`NeRFDecoder` (Standard MLP)。
  - 输入位置 $\to$ 输出密度 $\sigma$ 和特征向量。
  - 特征向量 + 观察方向 $\to$ 输出颜色 RGB。
- **输出**：该点的密度和视点相关的颜色。

### Part 2: Instant-NGP

**实验目标**：加速 NeRF 训练与渲染过程。

**核心原理**：

1.  **多分辨率哈希编码**：
    - 使用可学习的哈希表存储特征，替代固定的三角函数编码。
    - 使用哈希冲突（Hash Collision）来换取 $O(1)$ 的查询速度和巨大的参数空间。
    - 能够在极小的计算开销下捕捉极其精细的几何细节。
2.  **Density Grid**：
    - 维护一个粗糙的体素网格，标记空区域。
    - 光线投射时跳过空区域，大幅减少无效采样，提升效率。

**架构设计**：

- **编码层**：`HashRepresentation` (TinyCUDA-NN 实现)。
- **网络**：`InstantNeRFDecoder` (Tiny MLP)，通常只有 1-2 层隐含层，计算极快。
- **流程**：Hash Encodings $\to$ Tiny MLP $\to$ RGB/$\sigma$。
