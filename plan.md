# 1. 全局文件架构

我们的架构分为三层：**数据层 (Input)** -> **引擎层 (Processing)** -> **模型层 (Knowledge)**。数据单向流动，层级分明。

```text
src/
├── [模型层 - Model]       # 核心大脑：存储场景的几何与颜色信息
│   ├── abstract.py        # 接口定义：所有 Representation 和 Decoder 必须遵守的契约
│   ├── embeddings.py      # 坐标映射：负责将 (x,y,z) 映射为高维特征 (Fourier 或 Octree)
│   ├── decoders.py        # 信号解码：负责将特征映射为物理量 (RGB, Sigma)
│   └── core.py            # 组装工厂：组合 Embedding 和 Decoder 形成 NeuralField
│
├── [引擎层 - Engine]      # 物理引擎：负责光线传输与数值计算
│   └── renderer.py        # 渲染管线：统一管理 Sampling (采样), Compacting (稀疏化), Integration (积分)
│
├── [数据层 - Data]        # 数据入口：负责标准化输入
│   └── dataset.py         # 数据加载：负责读取磁盘文件，并进行坐标系归一化
│
├── [工具层 - Utils]
│   └── utils.py           # 辅助工具：PSNR 计算, 可视化, Checkpoint 管理
│
└── run.py                 # 训练总控：负责优化循环、验证逻辑与超参数管理
```

---

## 2. 详细模块设计与实现细节

### 2.1 数据层: `src/dataset.py`

**作用**: 数据搬运工与坐标系统治者。
**目的**: 无论原始数据是 Blender 还是 COLMAP，进入训练循环时，必须统一为标准化的射线，且物体必须位于 $[-1, 1]$ 的单位立方体内。

#### 类设计: `BlenderDataset`

- **内部逻辑**:

  1. **解析元数据**: 读取 `transforms.json`，获取相机内参 (`focal_length`) 和外参 (4x4 变换矩阵)。
  2. **生成光线 (Ray Generation)**:
     - 对图像每个像素 $(u, v)$，计算相机坐标系下的方向：
       $$
       x = (u - W/2)/f, \quad y = -(v - H/2)/f, \quad z = -1
       $$
     - 应用 `c2w` 矩阵旋转平移，得到世界坐标系下的 `rays_o` 和 `rays_d`。
  3. **坐标归一化 (关键)**:
     - Blender 合成数据集通常以原点为中心，但范围可能超过 $[-1, 1]$。
     - 计算 `self.scale` (例如 1.0/1.5)。
     - 变换 `rays_o`: $\mathbf{o}' = \mathbf{o} \times \text{scale}$。
     - **注意**: `rays_d` 保持模长为 1，不缩放，否则积分距离计算会出错。

- **输入输出 (I/O)**:

  - **Input**: `root_dir` (文件夹路径), `split` (train/val/test)。
  - **Output**: 字典或元组，包含 `rays_o` [N, H, W, 3], `rays_d` [N, H, W, 3], `target` [N, H, W, 3]。

- **接口预留**:

  - `self.near`, `self.far`: 根据缩放后的场景大小，固定导出 `near=2.0`, `far=6.0` (示例值)，供 Sampler 使用。
  - `self.bbox`: 返回场景的 AABB 包围盒，未来用于初始化 Octree 的根节点。
  - **注意**: 若做坐标缩放，`near/far` 必须与缩放一致；推荐基于 `bbox` 或分位数自动估计，避免固定值与场景尺度不匹配。
  - **Blender 背景**: 若存在 alpha 通道，需合成到白底或指定背景色，避免边缘变黑影响 PSNR。

---

### 2.2 引擎层: `src/renderer.py` (核心枢纽)

**作用**: 可微分渲染方程的求解器。
**目的**: 实现 **Compact-Batch (紧凑批处理)** 机制，一套代码同时支持稠密 MLP 和稀疏 Octree。

#### 核心组件 A: `StratifiedSampler`

- **内部逻辑**:

  - **分层采样**: 将 $[near, far]$ 分成 $N$ 个桶，每个桶内随机选一点（训练时）或取中点（测试时）。
  - **微小扰动**: 防止网络过拟合到特定的深度值。

- **输入输出**:

  - **Input**: 光线束 `rays_o`, `rays_d`, `near`, `far`。
  - **Output**: `z_vals` (深度), `pts` (3D 点), `mask` (有效性掩码)。

- **接口预留 & 稀疏化**:

  - **Vanilla 阶段**: `mask` 始终返回 `None`。
  - **（后续优化再说）Octree 阶段**: 新增 `OctreeSampler`，调用 `octree.query(pts)`，返回布尔 `mask`。仅当点在八叉树叶子节点内时为 True。

#### 核心组件 B: `render_rays` (流程控制器)

这是整个系统中最复杂、设计最精妙的函数。

1. 采样阶段

   首先，使用采样器（Sampler）在每条光线上生成一系列采样点。这一步会返回各点在光线上的深度值（z_vals）、三维空间坐标（pts），以及一个可选的掩码（mask）。如果处于全量模式（Vanilla），掩码为空；如果处于八叉树模式（Octree），掩码会标记出哪些点真正位于物体的叶子节点内。

2. 扁平化阶段

   为了方便后续统一处理，将所有光线（Rays）和每条光线上的采样点（Samples）这两个维度合并，把输入数据“拍扁”成一个长长的点列表。

3. 紧凑化阶段

   这是稀疏加速的关键。系统检查是否存在掩码：

   - 如果有掩码：提取掩码为 True 的有效点的索引，仅将这些有效点和对应的方向向量挑出来，组成一个更小的、紧凑的批次（Compact Batch）。这样可以直接剔除大量无效的空闲空间点，从而显著减少计算量。
   - 如果没有掩码：则使用全部点进行后续计算。

4. 推理阶段

   将上一步得到的点集送入神经场（Neural Field）模型进行前向传播。此时模型并不感知这些点属于哪条光线，它只是单纯地对这批点进行查询，输出对应的颜色（RGB）和密度（Sigma）。

5. 散射还原阶段

   将模型计算出的结果放回它们在原始列表中的位置：

   - 如果有掩码：创建一个与原始全量数据等大的全零容器，根据之前记录的有效索引，将计算结果填回对应的位置。那些被跳过的无效点在容器中保持为零（即无密度、无颜色）。
   - 如果没有掩码：结果本身就是全量的，无需特殊处理。

6. 积分计算

- **要求**:
  - 实现 `Early Ray Termination` (权重累积到 1.0 后停止采样)。

---

### 2.3 模型层: `src/core.py`, `embeddings.py`, `decoders.py`

**作用**: 定义场景的隐式表达。
**目的**: 实现高度解耦，使得更换编码方式（如从傅里叶变换改为八叉树查询）不需要修改解码器。

#### `abstract.py`

- **BaseRepresentation**: 定义 `out_dim` 属性，强制子类告知输出特征维度。
- **BaseDecoder**: 定义接收特征向量的标准接口。

#### `embeddings.py`

- **`FourierRepresentation` (Part 1/2)**:
  - **逻辑**: $\gamma(x) = [\sin(2^k \pi x), \cos(2^k \pi x)]_{k=0}^{L-1}$。
  - **目的**: 解决 MLP 的 "Spectral Bias" 问题，使其能学习高频细节。
  - **视角编码**: 视角方向 `d` 建议同样进行位置编码，并在 density head 后拼接进入 color head。
- **(后续优化再说) **` OctreeRepresentation`:
  - **逻辑**: 输入坐标 $x$ -> 计算 Morton Code -> 在八叉树 Hash 表中查询叶子节点 -> 三线性插值获取特征。
  - **优化**: 使用 `ocnn` 的 CUDA 算子实现极速查询。

#### `decoders.py`

- **`StandardMLP`**:
  - **结构**: 8 层深度，每层 256 宽。
  - **输入**: 自动根据 `representation.out_dim` 调整输入层。
  - **View Dependence**: 密度 $\sigma$ 仅依赖位置，颜色 RGB 依赖位置+视角。这意味着密度头 (Density Head) 先输出，然后特征向量与视角编码拼接后进入颜色头 (Color Head)。

#### `core.py` (`NeuralField`)

- **逻辑**: 作为一个容器 (`nn.Module`)，在 `__init__` 中根据配置实例化具体的 `Representation` 和 `Decoder`。
- **Forward**: 简单的 `x = rep(p); out = dec(x, d)` 流水线。

---

### 2.4 训练循环: `run.py`

**作用**: 整个系统的调度者。
**目的**: 最小化重建误差 (MSE Loss)，并定期评估各项指标。

- **逻辑流程**:

  1. **Setup**: 加载 Config，初始化 Dataset, Model, Optimizer。
  2. **Train Loop**:
     - 从 Dataset 获取一个 batch 的射线 (Batch Size 如 4096)。
     - 调用 `renderer.render_rays` 获取 `pred_rgb`。
     - 计算 loss = MSE(pred, target)。
     - `loss.backward()` & `optimizer.step()`。
  3. **Check & Grow (后续优化再说)**:
     - if `step % check_interval == 0`:
     - 检查是否是 Octree 模式。
     - 如果是，执行 `field.check_split()` (八叉树生长/细分)。
     - 如果树结构发生变化，执行 `optimizer = reset_optimizer(...)` (这也是接口预留的一部分)。

---

## 3. 执行路线图

### Step 1: 基础建设

- **Task 1.1**: 完善 `src/dataset.py`，确保 Blender 数据能正确加载且归一化。
- **Task 1.2**: 完善 `src/renderer.py`，现在只跑 `else` 分支。

### Step 2: 模型构建

- **Task 2.1**: 编写 `src/embeddings.py` 的傅里叶编码。
- **Task 2.2**: 编写 `src/decoders.py` 的 MLP，注意 Skip Connection 和 View Dependence 的正确实现。
- **Task 2.3**: 在 `src/core.py` 中将二者组装。

### Step 3: 训练与验证

- **Task 3.1**: 编写 `run.py`。
- **Task 3.2**: 在 Lego 数据集上先跑通稠密版本 (Vanilla NeRF)，目标是 50k steps 后 PSNR > 25db。
- **Task 3.3**: 新视角评估：渲染测试集并报告 PSNR/SSIM（或仅 PSNR），保证与任务要求对齐。
- **Task 3.4**: 生成环绕视角视频：给定相机轨迹（如球面/圆轨迹），渲染并输出视频用于定性展示。

### Step 4: 八叉树升级

- **Task 4.1**: 编写 `OctreeSampler` (利用 Octree 生成 mask)。
- **Task 4.2**: 编写 `OctreeRepresentation` (利用 OCNN 查特征)。
- **Task 4.3**: 在 `run.py` 中引入八叉树更新逻辑 (`grow`)。
- **Result**: 此时，渲染管线将自动利用 `valid_indices` 进行加速，无需改动 `renderer.py` 一行代码。
  - **可行性说明**: 该步骤依赖 CUDA/ocnn，工作量较大，可作为性能向扩展而非主线任务。
  - **可选替代**: 若需轻量加速，可先加入 occupancy/density grid 进行稀疏采样。
