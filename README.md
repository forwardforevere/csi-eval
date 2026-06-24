# CSIEval — CSI Feedback Compression Model Evaluation Framework

> A self-contained, pluggable evaluation framework for neural-network-based
> CSI feedback compression models used in 5G/6G wireless systems. Supports
> arbitrary model architectures via duck-typed protocols.

[![Python >= 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](#)
[![PyTorch >= 2.0](https://img.shields.io/badge/pytorch-2.0+-red.svg)](#)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Three Ways to Plug In a Model](#three-ways-to-plug-in-a-model)
- [Dataset Guide](#dataset-guide)
- [Configuration Reference](#configuration-reference)
- [Output API](#output-api)
- [Metric Reference](#metric-reference)
- [Custom Model Integration](#custom-model-integration)
- [Pre-Submission Checklist](#pre-submission-checklist)

---

## Quick Start

### Recommended Directory Structure

```
Test Project/              
├── csibench/                       
├── data/              
└── runs/                  
    └── best.pt
└── model/
    └── csinet.py
└── results/
└── main.py
```

### How to Use
```python
from csibench import Evaluator, EvalConfig

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

report["sgcs"]          # access single metric
report.save("html")     # write interactive HTML report
```

**Dataset path guide:**
- `data/2_6GHz` — Primary evaluation dataset (test set)
- `data/2_6GHz_part1_new` — Part 1 NEW (unseen environment, maps 1000+), for generalization evaluation
- `data/2_6GHz_part2` — Part 2 (dense deployment, scenario II), for generalization evaluation

At runtime, the `eig_cache/` subdirectory is automatically created under the dataset directory. This is managed by the framework and requires no user action.

**Requirements:**
```python
# Necessary dependency
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
# Optional dependencies
plotly>=5.0.0
pandas>=2.0.0
```


## Three Ways to Plug In a Model

### Method A — Pre-loaded model instance (recommended)

```python
import torch
from csibench import Evaluator, EvalConfig
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

This approach does not depend on model class name, layer names, or constructor signatures. Just pass the loaded model object.

### Method B — Checkpoint file only

```python
report = Evaluator(
    task="eigenvector_feedback",
    checkpoint="runs/best.pt",
    data="data/Dataset/wair_d_output/2_6GHz",
).run()
```

The framework builds a placeholder model (parameter shapes matching EVCsiNet Nt=32, K=13, reduction=8), then loads weights via `load_state_dict(strict=False)`.

### Method C — Model class + constructor arguments

```python
from csibench import Evaluator, EvalConfig

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

## Dataset Guide

### Dataset Download

You can download the preprocessed evaluation datasets from Hugging Face:

🔗 [https://huggingface.co/datasets/YSSAie/csi-eval-compression](https://huggingface.co/datasets/YSSAie/csi-eval-compression)

The archive contains the three ready-to-use subsets (`2_6GHz`, `2_6GHz_part1_new`, `2_6GHz_part2`) described below, already in the directory layout expected by `csibench`.

If you want to process the data yourself from raw recordings, use the data preparation scripts under [`data_generation_scripts/`](./data_generation_scripts) (see `scripts/generate_waird_csi_feedback_data.py`, `scripts/generate_part1_subset.py`, `scripts/generate_part2.py`) on the raw CSI data. The raw data source is:

🔗 [https://mobileai-dataset.cn/html/default/zhongwen/shujuji/1592719963402108929.html?index=1](https://mobileai-dataset.cn/html/default/zhongwen/shujuji/1592719963402108929.html?index=1)

### Directory Structure

```
data/              # ← user dataset root
├── 2_6GHz/                            # Primary evaluation dataset
│   ├── DATA_HtestI.npy             # Test set
│   └── eig_cache/                  # Framework-managed eigenvector cache
├── 2_6GHz_part1_new/              # Part 1 NEW (unseen deployment envs)
│   ├── DATA_HoodI.npy
│   └── eig_cache/
└── 2_6GHz_part2/                  # Part 2 (dense deployment, scenario II)
    ├── DATA_HallII.npy
    └── eig_cache/
```

### Data Preprocessing

During evaluation, the framework performs feature extraction (e.g., subband eigenvector computation) on the dataset, and caches results to the `eig_cache/` subdirectory within each dataset folder. This is fully automatic and requires no user intervention.




---

## Configuration Reference

### Required Parameters

| Parameter | Description |
|---|---|
| `task` | Task name. Built-in: `"eigenvector_feedback"` |
| `checkpoint` / `model` / `model_class` | One of three, provides the model |
| `data` | Path to the preprocessed dataset directory (where `DATA_Htest.npy` lives) |

### All Configurable Fields

#### Files & Paths

| Field | Default | Description |
|---|---|---|
| `data` | `data/Dataset/wair_d_output/2_6GHz` | Dataset path |
| `output_dir` | `results/eval` | Report output directory |
| `checkpoint` | `None` | Model checkpoint path |

#### Runtime

| Field | Default | Description |
|---|---|---|
| `device` | `"cuda"` | Compute device; auto-falls back to `"cpu"` if no GPU |
| `seed` | `42` | Random seed for reproducibility |

#### Metrics

| Field | Default | Description |
|---|---|---|
| `latency_runs` | `100` | Number of forward passes for latency measurement |
| `snr_levels_db` | `(5,10,15,20,25,30,40)` | SNR levels (dB) for noise robustness scan |
| `quant_bits_sweep` | `(0, 2, 4, 8)` | Bit-widths for quantization robustness scan |
| `report_formats` | `("json","html")` | Output formats |

#### Few-Shot Fine-Tuning

| Field | Default | Description |
|---|---|---|
| `fewshot_samples` | `(0,5,10,20,50,100,300)` | Support set sizes for each few-shot point |
| `fewshot_epochs` | `30` | Max fine-tuning epochs per few-shot point |
| `fewshot_lr` | `1e-4` | Fine-tuning learning rate |

#### Generalization Evaluation

| Field | Default | Description |
|---|---|---|
| `ood_targets` | `[]` | OOD target list |
| `include_default_ood` | `True` | Auto-register Part1_NEW and Part2 |

---

## Output API

```python
report = Evaluator(cfg).run()

report["sgcs"]                           # single metric
report["ood/part1_new::gap_nmse"]        # OOD sub-result
report.filter("task_performance")        # sub-report by category

report.save("json")       # results/eval/metrics.json
report.save("html")       # results/eval/report.html
report.save("markdown")   # results/eval/report.md

report.print_summary()    # print summary
```

---

## Metric Reference

### Task Performance

| Metric | Unit | Direction | Description |
|---|---|---|---|
| `nmse` | dB | ↓ lower is better | Normalized mean square error |
| `mse` | — | ↓ lower is better | Raw mean square error |
| `rho` | — | ↑ higher is better | Pearson correlation coefficient |
| `sgcs` | — | ↑ higher is better | Squared generalized cosine similarity |
| `evm` | dB | ↓ lower is better | Error vector magnitude |

### Deployment & Storage

| Metric | Unit | Direction | Description |
|---|---|---|---|
| `size` | MB | ↓ lower is better | On-disk checkpoint size |
| `params` | M | ↓ lower is better | Trainable parameter count |
| `memory` | MB | ↓ lower is better | Peak inference GPU memory |
| `bitwidth` | bits | ↓ lower is better | Quantization bit-width |
| `compression` | — | ↓ lower is better | Compression ratio (compressed/total) |
| `csi_reduction_rate` | % | ↑ higher is better | Feedback overhead reduction vs Rel-16/17 Type II codebook |
| `overhead` | bytes | ↓ lower is better | Over-the-air bytes per feedback |
| `loadtime` | s | ↓ lower is better | Checkpoint load time |

### Computation Efficiency

| Metric | Unit | Direction | Description |
|---|---|---|---|
| `latency` | ms | ↓ lower is better | Forward pass latency |
| `flops` | — | ↓ lower is better | FLOP count |
| `macs` | — | ↓ lower is better | MAC count |
| `traintime` | s | ↓ lower is better | Training wall-clock time |

### Robustness & Generalization

| Metric | Unit | Direction | Description |
|---|---|---|---|
| `snr_nmse` | — | — | NMSE slope vs SNR |
| `snr_sgcs` | — | ↑ higher is better | SGCS slope vs SNR |
| `quant` | dB | ↓ lower is better | NMSE at each quantization bit-width |
| `id_baseline` | dB | ↓ lower is better | In-distribution baseline (per-map mean NMSE) |
| `zero_shot` | dB | ↓ lower is better | Cross-scenario zero-shot NMSE (2-bit quantization) |
| `fine_tune` | % | ↑ higher is better | Few-shot fine-tuning recovery rate |
| `gap_nmse` | dB | ↓ lower is better | Gap = Zero-Shot − Baseline |
| `sgcs_decay_rate` | — | ↓ lower is better | SGCS relative decay rate |

---

## Custom Model Integration

A model only needs to implement two methods to integrate with the framework:

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

**Optional extensions** (implementing these enables richer metrics):

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

## Pre-Submission Checklist

When plugging in an **external model** (a checkpoint not trained by this repo, and/or not trained on the current dataset), confirm the following **before** running the evaluator. Any failure here makes the report numbers meaningless.

### 1. Dataset Shape Alignment
- `n_subcarriers`, `n_t`, and the complex-tensor convention (real/imag split vs complex-last) used during training must match this repo's config (see `csibench/core/config.py`).
- Run one `forward()` on a small batch and print the input shape to compare.

### 2. Checkpoint Compatibility
- `state_dict` must load with `strict=True` (key names and tensor shapes must match exactly).
- The checkpoint must contain a `model_info` field with at least `model_class` (with full `__module__` path) and `model_kwargs`.
- `model_info.model_class.__module__` must be importable from the current `sys.path`.

### 3. Input / Output Tensor Contract
- Input: `H_real, H_imag`, shape `(B, 2, n_subcarriers, n_t)`.
- Output: same shape `(B, 2, n_subcarriers, n_t)`. If your model outputs a different convention (e.g. `(n_subcarriers, n_t, 2)`), wrap it in a thin shim that transposes back.
- If using the evaluator's internal quantization wrapper, provide `quant_bits`.

### 4. Normalization Consistency
- The normalization used during training (mean / variance / clipping) must match the evaluation pipeline; otherwise NMSE will be inflated and the report misleading.

### 5. Metadata
- Provide `model_name` and `model_size_mb` (rendered as chips in the HTML report header).
- Provide the `task` name (used for report categorization).

### 6. Inference Consistency
- Use `model.eval()` and `torch.no_grad()`; freeze BN running stats.
- Keep device (CPU/CUDA) and `dtype` consistent.