# Project-NeRF

---

## 1. 资源说明

#### 项目网盘

[这里](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 存放了部分数据集和预训练模型，结果。

#### 项目报告

[milestone](https://latex.pku.edu.cn/project/6954fa6e57c9c512c3b063aa)

[report](https://latex.pku.edu.cn/7474895241mkhtrdjqsctt#529d39)

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

若要运行 **`Instant-NeRF`** ，请从 [网盘](https://disk.pku.edu.cn/anyshare/zh-cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B/83161E9603DD440884AEA38F67B808E2/84035BCADD4346ED9A22D6EA1AD00CEF/671028B590AA46C7BD1E45E6F4C4B0AF?_tb=none) 下载 `install_ngp.sh` 放置到项目文件夹运行一下，以确保所需的 TinyCUDA-NN 和相关组件正确安装。

#### 配置自定义

将 `configs/part{num}.yaml.example` **复制**一份并重命名为 `configs/part{num}.yaml`，并根据需要修改配置文件中的参数。

### 2.2 Part 1: 2D 图像拟合

**1. 数据准备**
将目标图像（如 `fox.jpg`）放入 `data/` 目录。可以使用网盘提供的测试图。

**2. 训练**

```bash
python3 run.py --image data/fox.jpg --config configs/part1.yaml
```

**3. 测试**

Part 1 训练过程中会自动进行拟合效果渲染。如果需要手动使用 checkpoint，请先从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载预训练模型，然后运行：

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

建议下载 NeRF Synthetic 数据集，放置在 `data/nerf_synthetic/`下，可以从 [网盘/part2/data](https://disk.pku.edu.cn/anyshare/zh-cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B/83161E9603DD440884AEA38F67B808E2/84035BCADD4346ED9A22D6EA1AD00CEF/671028B590AA46C7BD1E45E6F4C4B0AF/4D12B72493154D28819131CA3144D512/62BB1D321D0C4126889E8503E14E7FB7?_tb=none) 获取。

**2. 训练**

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml
```

**3. 测试**

请先从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载预训练模型并放置到 `output/part2/checkpoints/` 目录下，然后运行：

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml \
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

请先从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载预训练模型并放置到 `output/part2_instant/lego/` 目录下，然后运行：

```bash
# --render_n 参数：-1表示生成视频，正数表示随机渲染指定数量的视角
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2_instant.yaml \
  --checkpoint output/part2_instant/lego/best_model.pth --eval_only --render_n -1
```

**4. 输出说明**

- 结果保存在 `output/part2_instant/`。
- 使用 TensorBoard 监控损失下降与 PSNR 实时变化。
- 训练和渲染速度较快，可根据需要调整 [配置](configs/part2_instant.yaml) 中 `batch_size` 和 `chunk` 参数以适配显存。

### 2.5 Part 3: Dynamic NeRF (D-NeRF)

**Part 3 提供三种架构**：

- **Part 3 标准版** (`part3.yaml`)：Fourier MLP 变形场 + Fourier MLP 规范场
- **Part 3 DTC** (`part3_dtc.yaml`)：Direct Time Conditioning，无变形场，直接拼接时间编码
- **Part 3 Instant** (`part3_instant.yaml`)：Fourier MLP 变形场 + Hash 规范场（推荐）

**1. 数据准备**

使用 D-NeRF 数据集，可从 [网盘/part3/data](https://disk.pku.edu.cn/anyshare/zh-cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B/83161E9603DD440884AEA38F67B808E2/84035BCADD4346ED9A22D6EA1AD00CEF/671028B590AA46C7BD1E45E6F4C4B0AF/B9B86D05DF754D4B8A15C727136F93A0/4B4C851DDA53488CB79F8A943465ED6A?_tb=none) 获取并在 `data/` 下解压重命名为`d-nerf`。

**2. 训练**

```bash
# 推荐：Instant 版本（速度最快，效果好）
python3 run.py --data_dir data/d-nerf/standup --config configs/part3_instant.yaml

# 标准版本（收敛慢，需要更长训练时间）
python3 run.py --data_dir data/d-nerf/standup --config configs/part3.yaml

# Direct Time Conditioning（实验性架构，适合动作简单的场景，几乎无法收敛）
python3 run.py --data_dir data/d-nerf/standup --config configs/part3_dtc.yaml
```

**3. 测试**

请先从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载预训练模型并放置到 `output/part3/standup/` 目录下（对应不同配置的模型路径可能不同），然后运行：

```bash
# --render_n 参数：-1表示生成视频，正数表示随机渲染指定数量的视角
python3 run.py --data_dir data/d-nerf/standup --config configs/part3_instant.yaml \
  --checkpoint output/part3/standup/best_model.pth --eval_only --render_n -1
```

**4. 输出说明**

- 训练结果保存在 `output/part3/<dataset_name>`（标准/Instant）或 `output/part3_direct/<dataset_name>`（DTC）。
- 生成的视频会保存为 `<dataset_name>_24fps.mp4`。
- 支持环绕渲染（相机旋转 + 时间变化）。

### 2.6 Part 4: Dual-Hash Dynamic NeRF

**实验目标**：通过三网格时间锚点架构实现高质量动态场景重建。

**核心创新**：

- **Tri-Grid 时间锚点**：在 t=0, 0.5, 1 三个时刻使用独立 HashGrid，通过三角形加权插值实现 C1 连续
- **TV-Displacement Loss**：对位移网格施加全变分正则化，消除物体边缘闪烁
- **Static Anchor Loss**：强制 t=0 时零位移，确保规范空间有明确定义

**1. 数据准备**

使用与 Part 3 相同的 D-NeRF 数据集。

**2. 训练**

```bash
python3 run.py --data_dir data/d-nerf/standup --config configs/part4.yaml
```

**3. 测试**

请先从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 下载预训练模型并放置到 `output/part4/standup/` 目录下，然后运行：

```bash
# 只测试 PSNR
python3 run.py --data_dir data/d-nerf/standup --config configs/part4.yaml \
  --checkpoint output/part4/standup/best_model.pth --eval_only

# 生成环绕视频（300 帧）
python3 run.py --data_dir data/d-nerf/standup --config configs/part4.yaml \
  --checkpoint output/part4/standup/best_model.pth --eval_only --render_n -1
```

**4. 输出说明**

- 训练结果保存在 `output/part4/<dataset_name>`。
- 使用 `--eval_only` 模式会跳过视频渲染，仅输出测试集 PSNR。
- 视频文件命名为 `<dataset_name>_part4_24fps.mp4`。

---

## 3. 项目架构

```
Project-NeRF/
├── configs/                # 实验配置文件
│   ├── part1.yaml.example              # 2D 拟合配置模板
│   ├── part2.yaml.example              # 标准 NeRF 配置模板
│   ├── part2_instant.yaml.example      # Instant-NGP 配置模板
│   ├── part3.yaml.example              # Dynamic NeRF 标准版
│   ├── part3_instant.yaml.example      # Dynamic NeRF Instant 版（推荐）
│   ├── part3_dtc.yaml.example          # Dynamic NeRF Direct Time Conditioning
│   └── part4.yaml.example              # Dual-Hash Dynamic NeRF
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

### Part 3: Dynamic NeRF

**实验目标**：重建动态变化的 3D 场景（时间 $t$ 维度）。

**核心原理**：

1.  **变形场 (Deformation Field)**：
    - 假设物体仅发生几何形变，将动态空间的坐标 $(x, t)$ 映射到规范空间 (Canonical Space) 的坐标 $x_c$。
    - $$ x_c = x + \Delta x(x, t) $$
    - $\Delta x$ 由一个变形网络预测。
2.  **规范场 (Canonical NeRF)**：
    - 在规范空间中训练一个静态的 NeRF 模型。
    - 为了处理随时间变化的光影（非几何变化），规范网络的输入通常也会融合时间特征。

**架构设计（标准版 & Instant 版）**：

- **变形网络 (`DeformNet`)**：
  - 输入：位置 $x$ (Fourier 编码) + 时间 $t$ (Fourier 编码)。
  - 输出：位移向量 $\Delta x$。
- **规范网络 (`CanonicalNet`)**：
  - 输入：规范位置 $x_c = x + \Delta x$ (Fourier/Hash 编码) + 时间 $t$ + 观察方向 $d$。
  - 输出：$RGB, \sigma$。

**架构设计（Direct Time Conditioning）**：

- **无变形场**：直接将原始坐标 $x$、时间 $t$、方向 $d$ 的编码拼接输入 MLP
- **单一解码器**：
  - 输入：$[\text{embed}(x), \text{embed}(t), \text{embed}(d)]$
  - 输出：$RGB, \sigma$
- **优势**：架构简单，无需学习复杂的变形映射
- **局限**：难以处理大幅度几何变形，适合简单周期性动作（如关节旋转）

**技术创新**：

1.  **网络架构增强**：
    - **时变规范特征**：将时间编码特征直接拼接到规范空间特征中，增强模型对随时间变化的光影（如移动高光、阴影）的建模能力。
    - **混合架构**：结合 Instant-NGP HashGrid（规范场）与 MLP（变形场），在保持动态效果的同时大幅提升训练速度。

2.  **数据增强**：
    - **坐标噪声注入**：在训练时对变形网络的输入 $(x, t)$ 添加微小噪声 $(x \pm \epsilon)$，强制模型在邻域内输出相似位移，显著增强变形场的**空间平滑性**。
    - **随机背景**：训练时随机切换背景颜色（黑/白/噪点），防止模型利用背景过拟合，提高抠图质量。
    - **体积守恒约束**：对随机采样时刻的变形场施加约束，要求物体在不同时刻保持全局体积守恒，防止物体凭空膨胀或收缩。

3.  **正则化**：
    - **变形 L2 正则**：防止形变过大。
    - **TV Loss**：平滑 HashGrid 特征。
    - **时间平滑**：保证动作连贯。

### Part 4: Dual-Hash Dynamic NeRF

**实验目标**：通过全哈希架构实现动态场景的高效高质量重建。

**核心创新**：

1.  **Tri-Grid 时间锚点 (Tri-Grid Temporal Anchoring)**：
    - 使用三个独立的 HashGrid 分别记录 $t=0, 0.5, 1$ 时刻的位移场
    - 通过**三角形加权插值**（而非分段线性）实现 C1 连续的时间插值
    - 权重公式：$w_i = \max(0, 1 - |t - t_i| / 0.5)$，归一化后加权平均
    - **解决问题**：消除视频中 t=1/6, 1/2, 5/6 附近的卡顿现象

2.  **Dual-Hash 协同架构**：
    - **位移场**：三个 HashGrid（`deform_grid_start/mid/end`）+ 轻量 MLP 解码器
    - **规范场**：单个 HashGrid + Instant-NGP 解码器
    - **优势**：位移场和规范场都使用哈希表，训练速度快，表达能力强

3.  **TV-Displacement Loss**：
    - 对位移哈希网格施加全变分正则化：$\mathcal{L}_{TV} = \sum |p_i - p_{i+1}|$
    - **效果**：强制相邻哈希条目的位移向量相似，消除白色边缘闪烁和哈希碰撞噪点

4.  **Static Anchor Loss**：
    - 强制 $t=0$ 时刻 `deform_grid_start` 输出零位移
    - **保证规范空间有明确定义**（第一帧 = 规范姿态）
    - 消除底座抖动、避免规范空间漂移

**架构设计**：

- **时间编码 → 时间调制网络**：
  - 输入：时间 $t$ 的 Fourier 编码
  - 输出：时间调制向量（用于控制位移强度）
  - 设计：2 层轻量 MLP + Sigmoid 门控

- **三网格位移解码**：
  - 输入：空间坐标 $x$ → 三个 HashGrid 查询 → 插值融合 + 时间调制
  - 输出：位移向量 $\Delta x$

- **规范场渲染**：
  - 输入：规范位置 $x_c = x + \Delta x$ → HashGrid 编码 + 时间特征 + 方向
  - 输出：$RGB, \sigma$

**正则化策略**：

| 正则项          | 权重     | 作用                             |
| --------------- | -------- | -------------------------------- |
| TV-Displacement | 0.0001   | 消除位移网格的高频噪声和边缘闪烁 |
| TV-Canonical    | 0.000001 | 平滑规范空间哈希特征             |
| Deformation L2  | 0.0001   | 限制位移幅度，防止几何扭曲       |
| Temporal Smooth | 0.0001   | 强制相邻时刻位移连续             |
| Static Anchor   | 0.001    | t=0 时零位移约束                 |

**性能对比**：

- **训练速度**：比 Part 3 标准版快 5-10 倍
- **渲染质量**：与 Part 3 Instant 相当或更优（取决于正则化调优）
- **视频流畅度**：三角形插值显著改善时间连续性
