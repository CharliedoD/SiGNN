# [SiGNN: 脉冲诱导图神经网络](https://doi.org/10.1016/j.patcog.2024.111026)

[English](README.md)

SiGNN (Spike-induced Graph Neural Network) 是一个用于**动态图表示学习**的脉冲图神经网络框架。该模型结合了脉冲神经网络 (SNN) 的时序动态特性与图神经网络 (GNN) 的结构建模能力，通过多粒度时序聚合 (Multi-Granularity Temporal Aggregation) 机制实现对动态图上节点的高效表征学习。

## 框架图

![SiGNN Framework](figs/fw.png)

## 项目结构

```
SiGNN/
├── main.py                          # 训练入口：参数解析、训练/评估循环
├── setup.py                         # C++ 扩展编译脚本
├── signn/                           # 核心包
│   ├── __init__.py                  # 包导出
│   ├── model.py                     # SiGNN 模型主体
│   ├── layers.py                    # TALayer (时序感知层) 和 Aggregator (聚合器)
│   ├── neuron.py                    # BLIF 脉冲神经元及代理梯度函数
│   ├── utils.py                     # 工具函数 (种子设置、参数打印)
│   ├── datasets/                    # 数据集模块
│   │   ├── __init__.py
│   │   ├── base.py                  # Dataset 基类及通用工具
│   │   ├── dblp.py                  # DBLP 共著动态图
│   │   ├── tmall.py                 # Tmall 用户-商品交互图
│   │   └── patent.py                # US Patent 引用图
│   └── sampling/                    # 邻域采样模块
│       ├── __init__.py
│       ├── neighbor_sampler.py      # Sampler 和 RandomWalkSampler
│       └── sample_neighbor.cpp      # C++ 高性能邻域采样实现
├── data/                            # 数据目录 (需自行下载)
│   ├── dblp/
│   ├── tmall/
│   └── patent/
├── LICENSE
└── README.md
```

## 模块详细说明

### `signn/model.py` — SiGNN 模型

核心模型类 `SiGNN`，实现了完整的动态图节点表征学习流程：

- **初始化**：根据数据集的邻接矩阵构建采样器，创建多通道 TA 层和 1D 卷积池化层
- **编码 (`encode`)**：遍历所有时间步，在每个时间步上对目标节点进行多跳邻域采样和聚合，然后通过多粒度池化融合不同时间分辨率的嵌入
- **前向传播 (`forward`)**：调用编码器，返回节点表征的 logits

关键参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 动态图数据集实例 | — |
| `in_features` | 输入特征维度 | — |
| `out_features` | 输出类别数 | — |
| `hids` | 各层隐藏维度 | `[128, 64]` |
| `nchannels` | 时序聚合通道数 | `3` |
| `p` | 累积图采样比例 | `0.5` |
| `sizes` | 每层邻域采样大小 | `[5, 2]` |
| `surrogate` | 代理梯度函数 | `"arctan"` |

### `signn/layers.py` — 网络层

- **`Aggregator`**：邻域聚合层，包含两组线性变换：一组用于空间信号（经 Sigmoid 门控），一组用于时序信号（输入脉冲神经元）
- **`TALayer`**：时序感知层，将 Aggregator 和 BLIF 神经元串联，实现"聚合 → 脉冲发放 → 门控"的逐跳处理流程

### `signn/neuron.py` — 脉冲神经元

实现了多种代理梯度函数和 BLIF 神经元：

**代理梯度函数**（用于 SNN 反向传播时近似不可导的阶跃函数）：
| 名称 | 类 | 说明 |
|------|-----|------|
| `sigmoid` | `SigmoidSpike` | Sigmoid 导数作为代理梯度 |
| `triangle` | `TriangleSpike` | 三角形代理梯度 (Bellec et al. 2020) |
| `arctan` | `ArctanSpike` | Arctan 导数代理梯度 (Fang et al. 2020) |
| `mg` | `MultiGaussSpike` | 多高斯代理梯度 (Yin et al. 2021) |
| `super` | `SuperSpike` | SuperSpike 代理梯度 (Zenke et al. 2018) |

**BLIF 神经元**动态过程：
1. **充电 (Charge)**：`v = v - (v - v_reset) * τ + (1 - τ) * dv`
2. **双向发放 (Fire)**：检测正向和反向越阈
3. **重置 (Reset)**：发放后重置到 `v_reset`
4. **阈值更新 (Threshold Update)**：自适应阈值衰减

### `signn/datasets/` — 数据集

支持三个动态图基准数据集：

| 数据集 | 类 | 说明 | 时间步合并 |
|--------|-----|------|-----------|
| DBLP | `DBLP` | 合著关系动态图 | 无 |
| Tmall | `Tmall` | 用户-商品交互图，需节点重索引 | 每10步合并 |
| Patent | `Patent` | 专利引用网络 | 每2步合并 |

**`Dataset` 基类**提供：
- 特征读取 (`_read_feature`)
- 节点划分 (`split_nodes`)：支持分层采样的训练/验证/测试划分
- 边划分 (`split_edges`)：基于时间戳的边划分
- 时序快照迭代器 (`__getitem__`, `__iter__`)

### `signn/sampling/` — 邻域采样

- **`Sampler`**：基于 C++ 扩展的高性能邻域采样器，从 CSR 格式邻接矩阵中采样固定数量的邻居
- **`RandomWalkSampler`**：基于 `torch_cluster` 的随机游走采样器
- **`add_selfloops` / `eliminate_selfloops`**：邻接矩阵自环操作

### `signn/utils.py` — 工具函数

- `set_seed(seed)`：设置 NumPy、PyTorch (CPU/CUDA) 随机种子
- `tab_printer(args)`：以表格形式打印超参数配置

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- SciPy
- scikit-learn
- tqdm
- texttable

可选依赖（使用随机游走采样器时）：
- torch-cluster

### 编译 C++ 扩展

```bash
python setup.py build_ext --inplace
```

### 数据准备

将数据集文件放在 `./data/` 目录下，按以下结构组织：

```
data/
├── dblp/
│   ├── dblp.txt          # 边列表 (src dst timestamp)
│   ├── dblp.npy          # 节点特征
│   └── node2label.txt    # 节点标签
├── tmall/
│   ├── tmall.txt
│   ├── tmall.npy
│   └── node2label.txt
└── patent/
    ├── patent_edges.json  # 边列表 (JSON 格式)
    ├── patent_nodes.json  # 节点标签 (JSON 格式)
    └── patent.npy
```

## 使用方法

### 基本训练

```bash
# 在 DBLP 数据集上训练（默认参数）
python main.py --dataset DBLP

# 在 Tmall 数据集上训练
python main.py --dataset Tmall

# 在 Patent 数据集上训练
python main.py --dataset Patent
```

### 自定义超参数

```bash
python main.py \
    --dataset DBLP \
    --epochs 200 \
    --lr 0.005 \
    --hids 128 64 \
    --sizes 5 2 \
    --dropout 0.6 \
    --surrogate arctan \
    --alpha 1.0 \
    --p 0.5 \
    --nchannels 3 \
    --batch_size 1024 \
    --seed 2024 \
    --cuda cuda:0
```

### 命令行参数一览

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `DBLP` | 数据集名称 (DBLP/Tmall/Patent) |
| `--sizes` | int+ | `5 2` | 各层邻域采样大小 |
| `--hids` | int+ | `128 64` | 各层隐藏维度 |
| `--aggr` | str | `mean` | 聚合函数 (mean/sum) |
| `--sampler` | str | `sage` | 采样器类型 (sage/rw) |
| `--surrogate` | str | `arctan` | 代理梯度函数 |
| `--neuron` | str | `BLIF` | 脉冲神经元类型 |
| `--batch_size` | int | `1024` | 批量大小 |
| `--lr` | float | `0.005` | 学习率 |
| `--train_size` | float | `0.4` | 训练集比例 |
| `--alpha` | float | `1.0` | 代理梯度平滑因子 |
| `--p` | float | `0.5` | 累积图采样比例 |
| `--dropout` | float | `0.6` | Dropout 概率 |
| `--epochs` | int | `100` | 训练轮数 |
| `--concat` | flag | `False` | 是否拼接自身与邻域表示 |
| `--seed` | int | `2024` | 随机种子 |
| `--nchannels` | int | `3` | 时序聚合通道数 |
| `--cuda` | str | `cuda:0` | CUDA 设备 |
| `--invth` | float | `1.0` | 初始电压阈值 |

### 作为 Python 包使用

```python
from signn import SiGNN, DBLP, set_seed

# 加载数据
data = DBLP(root="./data")
data.split_nodes(train_size=0.35, val_size=0.05, test_size=0.6)

set_seed(2024)

# 构建模型
import torch
device = torch.device("cuda:0")

model = SiGNN(
    dataset=data,
    in_features=data.num_features,
    out_features=data.num_classes,
    hids=[128, 64],
    sizes=[5, 2],
    surrogate="arctan",
    device=device,
).to(device)

# 前向传播
nodes = torch.arange(100)
logits = model(nodes)
```

## 许可证

本项目基于 MIT 许可证发布，详见 [LICENSE](LICENSE)。
