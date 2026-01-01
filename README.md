## 说明

#### 1.项目网盘：

[这里](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 存放了部分数据集和预训练模型。

#### 2.项目报告：

[milestone](https://latex.pku.edu.cn/project/6954fa6e57c9c512c3b063aa)

## 快速开始

#### 1. 环境配置

```bash
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. 准备数据

将您的图像放入 `data/` 目录，例如 `data/fox.jpg`，可从[网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B)获取。

#### 3. 运行 Part 1：2D 图像拟合

```bash
python3 run.py --image data/fox.jpg --config configs/part1_fox.yaml
```

训练完成后，结果会保存到 output/result_part1.png

终端会输出最终的 PSNR 值

#### 4. 实验参数调整

编辑 `configs/part1_fox.yaml` 来尝试不同配置：

**实验建议**：

- **无位置编码**: 将 `L_embed` 设为 0（需修改代码）
  - _原因_：对比基线，验证位置编码的必要性
  - _现象_：无编码时图像会模糊，无法捕捉高频细节（如边缘、纹理）
- **不同频率**: 尝试 `L_embed = 5, 10, 15`
  - _原因_：L 控制频率范围，影响能表示的细节层次
  - _现象_：L 太小 → 细节丢失；L 太大 → 可能过拟合/训练不稳定
- **网络深度**: 修改 `decoders.py` 中的层数
  - _原因_：更深网络有更强表达能力，但训练更困难
  - _建议_：尝试 2 层、3 层（当前）、5 层
- **隐藏层大小**: 尝试 `hidden_dim = 128, 256, 512`
  - _原因_：控制网络容量，影响拟合能力和训练速度
  - _权衡_：小 → 训练快但可能欠拟合；大 → 效果好但训练慢

#### 5. 查看 PSNR 指标

PSNR (Peak Signal-to-Noise Ratio) 是图像质量评估指标，数值越高表示重建质量越好：

- **< 30 dB**: 质量较差
- **30-40 dB**: 质量良好
- **> 40 dB**: 质量优秀

## 项目架构

```
OctNeRF/
├── configs/                # YAML 配置文件
│   └── part1_fox.yaml
├── data/
│   └── fox.jpg
├── output/
│   └── result_part1.png
├── src/
│   ├── __init__.py
│   ├── abstract.py         # 定义所有接口 (基类)
│   ├── embeddings.py       # 【组件】坐标映射 (Fourier/Octree)
│   ├── decoders.py         # 【组件】特征解码 (MLP)
│   ├── core.py             # 【引擎】组装组件
│   └── utils.py            # 工具函数
├── run.py                  # [入口] 统一启动脚本
├── requirements.txt        # 依赖列表
└── README.md               # 项目文档
```

## 核心概念

#### 为什么需要位置编码？

**问题**：标准 MLP 存在 **频谱偏差 (Spectral Bias)**

- 神经网络天然倾向于学习**低频函数**（平滑变化）
- 对于高频信息（图像的边缘、纹理等细节）学习困难
- 直接用坐标 $(x, y)$ 输入 MLP，会得到模糊的结果

**解决方案**：位置编码（Positional Encoding）

- 将低维坐标映射到高维频率空间
- 让网络能够"看到"不同频率的信息
- 这是 NeRF 能够合成高质量图像的关键技术

#### 位置编码 (Positional Encoding)

对于输入坐标 $p$，使用 Fourier 特征映射：

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), ..., \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

- **输入**：1D 坐标 $p \in \mathbb{R}$
- **输出维度**：$2L$（L 个频率，每个产生 sin 和 cos）
- **对于 2D**：$(x, y)$ 编码后维度为 $2 + 4L$（原始坐标 + 编码）
- **频率范围**：$[2^0\pi, 2^{L-1}\pi]$，覆盖从低频到高频

#### 架构设计

```
输入坐标 (x, y)
    ↓
[位置编码层] FourierRepresentation
    ↓
特征向量 (dim = 4L)
    ↓
[MLP 解码层] StandardMLP
    ↓
输出 RGB (3 通道)
```

## 实验要求

**Part 1 - 2D 图像拟合**：

- 实现位置编码
- 实现标准 MLP
- 计算 PSNR 指标
- **对比实验**（需要自己进行）：
  - **有 vs 无位置编码**：验证编码的重要性
  - **不同频率数 L**：观察频率对细节的影响
  - **不同网络深度**：3 层 vs 5 层 vs 8 层
  - **不同隐藏维度**：128 vs 256 vs 512
