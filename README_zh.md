# CSIEval — CSI 反馈压缩模型评测框架

> 用于 5G/6G 无线系统中神经网络 CSI（信道状态信息）反馈压缩模型的可插拔自包含评测框架。通过 duck-typed 协议支持任意模型架构。

---

## 目录

- [快速上手](#快速上手)
- [三种模型接入方式](#三种模型接入方式)
- [数据说明](#数据说明)
- [配置参考](#配置参考)
- [输出接口](#输出接口)
- [可用指标说明](#可用指标说明)
- [自定义模型接入](#自定义模型接入)
- [提交前自检清单](#提交前自检清单)

---

## 快速上手

### 推荐目录结构

```
Test Project/              
├── csieval/                       
├── data/              
└── runs/                  
    └── best.pt
└── model/
    └── csinet.py
└── results/
└── main.py
```
### 如何使用
```python
from csieval import Evaluator, EvalConfig

report = Evaluator(
        task="eigenvector_feedback",
        model="model/ev_csinet.py",
        checkpoint="runs/best.pt",
        data="data/2_6GHz",
        output_dir="results/test",
    ).run()

report.print_summary()
# [↓] nmse: -17.3919
# [↑] sgcs: 0.9811
# ...

report["sgcs"]          # 访问单个指标
report.save("html")     # 写入交互式 HTML 报告
```

**数据集路径说明**：
- `data/2_6GHz` — 基础评测数据集（测试集）
- `data/2_6GHz_part1_new` — Part 1 NEW（未见环境，地图 1000+），用于泛化评估
- `data/2_6GHz_part2` — Part 2（密集部署，场景 2），用于泛化评估

运行时，`eig_cache/` 子目录会在数据集目录下自动生成，是框架运行时自动管理的缓存，无需用户操作。

**依赖需求:**
```python
# Necessary dependency
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
# Optional dependencies
plotly>=5.0.0
pandas>=2.0.0
```

## 三种模型接入方式

### 方式 A — 直接传入已加载的模型实例（推荐）

```python
import torch
from csieval import Evaluator, EvalConfig
from mymodel import MyCsiNet

my_model = MyCsiNet(nt=32, n_subbands=13)
my_model.load_state_dict(torch.load("runs/best.pt"))
my_model.eval()

cfg = EvalConfig(
    task="eigenvector_feedback",
    model=my_model,
    checkpoint="runs/best.pt",   
    data="data/2_6GHz",
)
report = Evaluator(cfg).run()
```
此方案不依赖模型类名、层名或构造函数签名。只需传入已加载的模型对象即可。

### 方式 B — 仅提供 checkpoint 文件

```python
report = Evaluator(
    task="eigenvector_feedback",
    checkpoint="runs/best.pt",
    data="data/Dataset/wair_d_output/2_6GHz",
).run()
```

框架构建一个占位模型（参数形状与 EVCsiNet Nt=32, K=13, reduction=8 匹配），然后 `load_state_dict(strict=False)` 加载权重。

### 方式 C — 传入模型类 + 构造函数参数

```python
from csieval import Evaluator, EvalConfig

cfg = EvalConfig(
    task="eigenvector_feedback",
    checkpoint="runs/best.pt",
    model_class=MyCsiNet,
    model_kwargs={
        "nt": 32,
        "n_subbands": 13,
        "compression_dim": 104,
    },
    data="data/Dataset/wair_d_output/2_6GHz",
)
report = Evaluator(cfg).run()
```

---

## 数据说明

### 数据集下载

可直接从 Hugging Face 下载预处理好的评测数据集：

🔗 [https://huggingface.co/datasets/YSSAie/csi-eval-compression](https://huggingface.co/datasets/YSSAie/csi-eval-compression)

压缩包内已包含下面介绍的三份子集（`2_6GHz`、`2_6GHz_part1_new`、`2_6GHz_part2`），目录结构与 `csieval` 期望的一致，下载后即可直接使用。

如果希望自行从原始数据处理，可使用 [`data_generation_scripts/`](./data_generation_scripts) 目录下的数据处理脚本（`scripts/generate_waird_csi_feedback_data.py`、`scripts/generate_part1_subset.py`、`scripts/generate_part2.py`）在 Raw 数据上处理。原始数据来源：

🔗 [https://mobileai-dataset.cn/html/default/zhongwen/shujuji/1592719963402108929.html?index=1](https://mobileai-dataset.cn/html/default/zhongwen/shujuji/1592719963402108929.html?index=1)

### 数据目录结构

```
data/              # ← 用户数据集根目录
├── 2_6GHz/                            # 基础评测数据集
│   ├── DATA_HtestI.npy             # 测试集
│   └── eig_cache/                  # 框架运行时自动生成的特征缓存
├── 2_6GHz_part1_new/              # Part 1 NEW（未见部署环境）
│   ├── DATA_HoodI.npy
│   └── eig_cache/
└── 2_6GHz_part2/                  # Part 2（密集部署，场景 2）
    ├── DATA_HallII.npy
    └── eig_cache/
```

### 数据预处理

评测时，框架对数据集做特征提取（如子带特征向量计算），结果自动缓存到各数据集目录下的 `eig_cache/` 子目录。这是框架自动管理的，无需用户干预。

---

## 配置参考

### 必需参数

| 参数 | 说明 |
|---|---|
| `task` | 任务名称。内置：`"eigenvector_feedback"` |
| `checkpoint` / `model` / `model_class` | 三选一，提供模型 |
| `data` | 预处理后的数据集路径（即 `DATA_Htest.npy` 所在目录） |

### 全部可配置字段

#### 文件与路径

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data` | `data/Dataset/wair_d_output/2_6GHz` | 数据集路径 |
| `output_dir` | `results/eval` | 报告文件输出目录 |
| `checkpoint` | `None` | 模型 checkpoint 路径 |

#### 运行时

| 字段 | 默认值 | 说明 |
|---|---|---|
| `device` | `"cuda"` | 计算设备，无 GPU 时自动降级为 `"cpu"` |
| `seed` | `42` | 随机种子，保证可复现性 |

#### 指标相关

| 字段 | 默认值 | 说明 |
|---|---|---|
| `latency_runs` | `100` | 延迟测量的前向传播次数 |
| `snr_levels_db` | `(5,10,15,20,25,30,40)` | 噪声鲁棒性扫描的 SNR 级别 (dB) |
| `quant_bits_sweep` | `(0, 2, 4, 8)` | 量化鲁棒性扫描的位宽 |
| `report_formats` | `("json","html")` | 输出格式 |

#### 少样本微调

| 字段 | 默认值 | 说明 |
|---|---|---|
| `fewshot_samples` | `(0,5,10,20,50,100,300)` | 各少样本点的支撑集大小 |
| `fewshot_epochs` | `30` | 每个少样本点的最大微调轮数 |
| `fewshot_lr` | `1e-4` | 微调学习率 |

#### 泛化评估

| 字段 | 默认值 | 说明 |
|---|---|---|
| `ood_targets` | `[]` | OOD 目标列表 |
| `include_default_ood` | `True` | 自动注册 Part1_NEW 和 Part2 |

---

## 输出接口

```python
report = Evaluator(cfg).run()

report["sgcs"]                           # 单个指标
report["ood/part1_new::gap_nmse"]        # OOD 子结果
report.filter("task_performance")          # 某一类别的子报告

report.save("json")       # results/eval/report.json
report.save("html")       # results/eval/report.html
report.save("markdown")   # results/eval/report.md

report.print_summary()    # 打印摘要
```

---

## 可用指标说明

### 任务性能

| 指标 | 单位 | 方向 | 说明 |
|---|---|---|---|
| `nmse` | dB | ↓ 越小越好 | 归一化均方误差 |
| `mse` | — | ↓ 越小越好 | 原始均方误差 |
| `rho` | — | ↑ 越大越好 | Pearson 相关系数 |
| `sgcs` | — | ↑ 越大越好 | 广义余弦相似度平方 |
| `evm` | dB | ↓ 越小越好 | 误差向量幅度 |

### 部署存储

| 指标 | 单位 | 方向 | 说明 |
|---|---|---|---|
| `size` | MB | ↓ 越小越好 | 磁盘 checkpoint 大小 |
| `params` | M | ↓ 越小越好 | 可训练参数量 |
| `memory` | MB | ↓ 越小越好 | 推理峰值显存 |
| `bitwidth` | bits | ↓ 越小越好 | 量化位宽 |
| `compression` | — | ↓ 越小越好 | 压缩比（compressed/total） |
| `csi_reduction_rate` | % | ↑ 越大越好 | 相对于 Rel-16/17 Type II 码本的反馈开销降低率 |
| `overhead` | bytes | ↓ 越小越好 | 每次反馈的空中传输字节数 |
| `loadtime` | s | ↓ 越小越好 | checkpoint 加载时长 |

### 计算效能

| 指标 | 单位 | 方向 | 说明 |
|---|---|---|---|
| `latency` | ms | ↓ 越小越好 | 前向传播延迟 |
| `flops` | — | ↓ 越小越好 | FLOP 计数 |
| `macs` | — | ↓ 越小越好 | MAC 计数 |

### 鲁棒泛化

报告分三节：

- **4.1 Noise Robustness**：SNR-NMSE/SGCS 斜率曲线
- **4.2 Quantization Robustness**：各量化位宽下的 NMSE/SGCS 曲线
- **4.3 Cross-Scenario Evaluation**（需配置 OOD 目标）：跨场景零样本/微调对比表格 + 少样本曲线

Cross-Scenario Evaluation 包含：Per-Map NMSE/SGCS 均值与标准差、Gap NMSE、SGCS Generalized Decay Rate（2-bit 量化）。

---

## 自定义模型接入

模型只需实现两个方法即可接入框架：

```python
import torch.nn as nn

class MyCsiNet(nn.Module):
    def __init__(self, nt=32, n_subbands=13, compression_dim=104):
        super().__init__()
        self.nt = nt
        self.n_subbands = n_subbands
        self._input_shape = (2, nt, n_subbands)   # (C, H, W)
        self.compression_dim = compression_dim
        self.quant_bits = 0

        self.encoder = nn.Sequential(
            nn.Linear(2 * nt * n_subbands, 4 * compression_dim),
            nn.ReLU(),
            nn.Linear(4 * compression_dim, compression_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, 4 * compression_dim),
            nn.ReLU(),
            nn.Linear(4 * compression_dim, 2 * nt * n_subbands),
        )

    def forward(self, x):
        # x: [B, 2, Nt, K]
        b, c, nt, k = x.shape
        h = self.encoder(x.reshape(b, -1))
        h = self.decoder(h)
        return h.reshape(b, c, nt, k)

    def get_input_shape(self):
        return self._input_shape
```

**可选扩展**（实现后可获得更丰富的指标）：

```python
    def encode(self, x):
        b = x.shape[0]
        return self.encoder(x.reshape(b, -1))

    def get_compression_ratio(self):
        total = 2 * self.nt * self.n_subbands
        return self.compression_dim / total

    def get_quant_bits(self):
        return self.quant_bits
```

---

## 提交前自检清单

当你要把一个**外部模型**（非本仓库训练、未在当前数据集训练过的 checkpoint）接入 `csieval` 评估时，请先确认满足以下条件。任一项不满足，评估结果都不可信。

### 1. 数据集形状对齐
- 确认训练时使用的 `n_subcarriers`、`n_t`、复数表达方式（实/虚分两通道 vs 复数末维）与本仓库配置一致（见 `csieval/core/config.py`）。
- 在小批量上 `forward()` 一次，打印输入张量形状用于对比。

### 2. checkpoint 兼容
- `state_dict` 能被 `strict=True` 加载（key 命名、shape 完全一致）。
- checkpoint 内含 `model_info` 字段，至少有 `model_class`（含完整 `__module__` 路径）和 `model_kwargs`。
- `model_info.model_class.__module__` 必须在当前 Python 路径上能被 `import` 到。

### 3. 输入/输出张量约定
- 输入：`H_real, H_imag`，形状 `(B, 2, n_subcarriers, n_t)`。
- 输出：与输入同形状 `(B, 2, n_subcarriers, n_t)`；若模型输出其它约定（如 `(n_subcarriers, n_t, 2)`），需在 shim 中转置。
- 若使用评估器内部量化 wrapper，需提供 `quant_bits`。

### 4. 归一化一致性
- 训练时的归一化方式（均值/方差/clipping）必须与评估流水线一致；否则 NMSE 会被严重放大、报告失真。

### 5. 元信息
- 提供 `model_name`、`model_size_mb`（用于 HTML 报告头部 chip）。
- 提供 `task` 名（用于报告分类）。

### 6. 推理一致性
- 评估时 `model.eval()`、使用 `torch.no_grad()`、关闭 BN running stats 更新。
- 设备一致（CPU/CUDA），`dtype` 一致。

