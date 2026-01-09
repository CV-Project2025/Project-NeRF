## 资源说明

#### 1.项目网盘：

[这里](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 存放了部分数据集和预训练模型。

#### 2.项目报告：

[milestone](https://latex.pku.edu.cn/project/6954fa6e57c9c512c3b063aa)

## 快速开始

#### 1. 克隆代码库

```bash
git clone https://github.com/CV-Project2025/Project-NeRF
cd Project-NeRF
```

#### 2. 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. 准备数据（Part 1）

将您的图像放入 `data/` 目录，例如 `data/fox.jpg`，可从 [网盘](https://disk.pku.edu.cn/link/AAF9F7FDBF0428495A933F2CAFB52E944B) 获取。

#### 4. 直接运行 Part 1：2D 图像拟合

```bash
python3 run.py --image data/fox.jpg --config configs/part1.yaml
```

- 运行 `python3 run.py -h` 查看参数说明

- 训练完成后，中间结果（调整 [配置](configs/part1.yaml) 里面的`save_every`）会保存到 `output/runs/<run_name>/steps/`，最终结果会保存到 `output/runs/<run_name>/final.png`

- 终端会输出最终的 PSNR 值（**< 30 dB**: 质量较差， **30-40 dB**: 质量良好， **> 40 dB**: 质量优秀）

#### 5. 实验参数调整（Part 1）

**编辑 [configs/part1.yaml](configs/part1.yaml) 进行参数调整**，配置文件包含：

- **位置编码开关**：`use_positional_encoding: true/false`
- **频率数量**：`L_embed: 5, 10, 15, 20`
- **隐藏层维度**：`hidden_dim: 128, 256, 512, 1024`
- **网络深度**：`num_layers: 2, 3, 5, 8`
- **训练参数**：`epochs`, `learning_rate`

**推荐实验组合**：

1. **验证位置编码**：对比 `use_positional_encoding: false` vs `true`
2. **频率扫描**：测试 `L_embed = [5, 10, 15, 20]`
3. **网络容量**：测试 `hidden_dim = [128, 256, 512]`
4. **深度实验**：修改 `num_layers = [2, 3, 5, 8]` 测试不同层数

#### 6. 准备数据（Part 2）

本项目支持 NeRF Synthetic 数据格式，建议放置在 `data/nerf_synthetic/` 下，例如：

```
data/nerf_synthetic/lego/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
├── val/
└── test/
```

> `transforms_*.json` 内的 `file_path` 通常形如 `./train/r_0`，程序会自动补全扩展名（`.png/.jpg`）。

#### 7. 运行 Part 2：多视角 NeRF

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml
```

- 训练输出默认保存在 `output/part2/` 目录
- 测试集渲染结果位于 `output/part2/renders/`
- 若数据集中没有 `transforms_test.json`，程序会自动使用 `val` 集合进行评估
- 若显存不足，可在运行时指定更小的渲染块大小，例如 `--render_chunk 4096`

#### 8. 使用已有模型进行测试（Part 2）

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2.yaml \
  --checkpoint output/part2/checkpoints/model_step_020000.pth --eval_only
```

#### 9. 质量优先配置（Part 2，可选）

```bash
python3 run.py --data_dir data/nerf_synthetic/lego --config configs/part2_quality.yaml
```

- `configs/part2_quality.yaml` 提高训练步数与采样数，通常能获得更高 PSNR（训练更慢）
- 若需生成视频，可用 `ffmpeg` 将测试集渲染结果拼成视频，例如：

```bash
ffmpeg -framerate 30 -i output/part2/renders/test_%03d.png -c:v libx264 -pix_fmt yuv420p output/part2/nerf_video.mp4
```

## 项目架构

```
OctNeRF/
├── configs/                # YAML 配置文件
│   ├── part1.yaml
│   └── part2.yaml
├── data/
│   └── fox.jpg
├── output/
│   ├── runs/
│   └── part2/
├── src/
│   ├── __init__.py
│   ├── abstract.py         # 定义所有接口 (基类)
│   ├── dataset.py          # 数据加载 (Blender/NeRF Synthetic)
│   ├── embeddings.py       # 【组件】坐标映射 (Fourier/Octree)
│   ├── decoders.py         # 【组件】特征解码 (MLP)
│   ├── core.py             # 【引擎】组装组件
│   ├── renderer.py         # 体渲染与采样
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
