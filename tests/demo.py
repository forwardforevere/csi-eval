"""End-to-end demo: run the full evaluation framework on a small dummy model.

This script:
  1. Generates a tiny random "eigenvector" dataset in memory (no real
     WAIR-D data needed; uses the same layout [B, 2, Nt, K]).
  2. Trains a tiny EVCsiNet-style model for 2 epochs to get non-trivial
     weights (so compression / FLOPs / quantization all have meaning).
  3. Runs ``Evaluator.run()`` over all 4 metric categories and produces
     JSON + HTML + Markdown reports in ``results/ef_demo/``.

Run:
  python -m csieval.tests.demo
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. A tiny "fake" dataset that mimics WAIREigenDataset's [2, Nt, K] layout
# ---------------------------------------------------------------------------

class FakeEigenDataset(Dataset):
    """Random [2, Nt, K] tensors that look like normalized eigenvectors.

    We sample a complex unit-norm vector per (sample, subband) and add a
    small amount of noise so that a real model has something to fit.
    """

    def __init__(self, n_samples: int = 256, nt: int = 32, n_subbands: int = 13,
                 noise: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        K = n_subbands
        re = rng.standard_normal((n_samples, nt, K)).astype(np.float32)
        im = rng.standard_normal((n_samples, nt, K)).astype(np.float32)
        x = re + 1j * im
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        x = x + noise * (rng.standard_normal(x.shape).astype(np.complex64) +
                         1j * rng.standard_normal(x.shape).astype(np.complex64))
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        # Real/imag layout: [N, 2, Nt, K] (paper)
        self.x = torch.from_numpy(np.stack([x.real, x.imag], axis=1).astype(np.float32))
        # Also keep the complex form so DataAdapter.get_complex_raw works.
        self.complex = x.astype(np.complex64)
        # Fake "env" structure: 5 BS * 30 UE = 150 samples per map
        self.samples_per_env = 150
        # Fake config_info for the WAIREigenDataset contract
        self.config_info = {
            "task": "eigenvector_feedback", "config": "2_6GHz",
            "nt": nt, "nr": 4, "nf": 104, "n_subbands": K, "subband_size": 8,
            "n_samples": n_samples, "phase_fix": True, "use_cache": True,
        }
        self.shape = list(self.x.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# ---------------------------------------------------------------------------
# 2. A tiny EVCsiNet-style model with all optional capabilities
# ---------------------------------------------------------------------------

class TinyEVCsiNet(nn.Module):
    """Minimal mirror of EVCsiNet, exposing all the optional hooks."""

    def __init__(self, nt: int = 32, n_subbands: int = 13, embed_dim: int = 32,
                 num_layers: int = 2, reduction: int = 4, quant_bits: int = 0):
        super().__init__()
        self.nt = nt
        self.n_subbands = n_subbands
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.reduction = reduction
        self.quant_bits = quant_bits
        self.total_dim = 2 * nt * n_subbands
        self.seq_len = n_subbands
        self.patch_dim = 2 * nt
        self.compressed_dim = max(1, (self.seq_len * embed_dim) // reduction)
        self.input_shape = (2, nt, n_subbands)

        self.expansion = nn.Linear(self.total_dim, self.seq_len * embed_dim)
        self.contraction = nn.Linear(self.seq_len * embed_dim, self.total_dim)
        self.fc_enc = nn.Linear(self.seq_len * embed_dim, self.compressed_dim)
        self.fc_dec = nn.Linear(self.compressed_dim, self.seq_len * embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode(self, x):
        B = x.shape[0]
        if x.shape[2] < x.shape[3]:
            x = x.transpose(2, 3)
        flat = x.reshape(B, -1).float()
        tokens = self.expansion(flat).view(B, self.seq_len, self.embed_dim)
        tokens = self.encoder(tokens)
        code = self.fc_enc(tokens.reshape(B, -1))
        if self.quant_bits > 0:
            levels = (1 << self.quant_bits) - 1
            x_min = code.amin(dim=1, keepdim=True)
            x_max = code.amax(dim=1, keepdim=True)
            scale = (x_max - x_min).clamp_min(1e-8)
            code = (torch.round((code - x_min) / scale * levels) / levels) * scale + x_min
        return code

    def decode(self, code):
        B = code.shape[0]
        tokens = self.fc_dec(code).view(B, self.seq_len, self.embed_dim)
        tokens = self.decoder(tokens)
        flat = self.contraction(tokens.reshape(B, -1))
        return flat.view(B, 2, self.nt, self.n_subbands)

    def forward(self, x):
        if x.shape[2] < x.shape[3]:
            x = x.transpose(2, 3)
        return self.decode(self.encode(x))

    def get_compression_ratio(self):
        return self.compressed_dim / self.total_dim

    def get_input_shape(self):
        return self.input_shape

    def estimate_macs(self, batch_size: int = 1) -> int:
        D, FF, L, H = self.embed_dim, 128, self.seq_len, 4
        head_dim = D // H
        patch_mac = batch_size * L * self.patch_dim * D
        attn_mac = batch_size * self.num_layers * (
            2 * L * D * D + L * L * D + L * L * head_dim * H
        )
        ffn_mac = batch_size * self.num_layers * (2 * L * D * FF)
        bn_mac = batch_size * (self.total_dim * self.compressed_dim +
                                self.compressed_dim * self.total_dim)
        return int(patch_mac * 2 + (attn_mac + ffn_mac) * 2 + bn_mac)

    def get_model_info(self):
        params = sum(p.numel() for p in self.parameters())
        return {
            "name": f"TinyEVCsiNet_Nt{self.nt}_K{self.n_subbands}",
            "type": "TinyEVCsiNet",
            "task": "eigenvector_feedback",
            "input_shape": list(self.input_shape),
            "nt": self.nt, "n_subbands": self.n_subbands,
            "total_dim": self.total_dim,
            "compressed_dim": self.compressed_dim,
            "compression_ratio": self.get_compression_ratio(),
            "reduction": self.reduction,
            "params": params,
            "params_m": params / 1e6,
            "model_size_mb": params * 4 / (1024 ** 2),
            "quant_bits": self.quant_bits,
        }


# ---------------------------------------------------------------------------
# 3. A DataAdapter that wraps the FakeEigenDataset
# ---------------------------------------------------------------------------

class FakeEigenAdapter:
    def __init__(self, n_train=256, n_test=128, nt=32, K=13, seed=0):
        self._train = FakeEigenDataset(n_train, nt, K, noise=0.05, seed=seed)
        self._test = FakeEigenDataset(n_test, nt, K, noise=0.05, seed=seed + 1)
        self.samples_per_env = 150
        self.n_bs_per_map = 5
        self.n_ue_per_map = 30
        self._nt = nt
        self._K = K
        self._cfg = "2_6GHz"

    def get_loader(self, split, batch_size=128, num_workers=0, shuffle=False):
        ds = self._train if split == "train" else self._test
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_complex_raw(self, split):
        ds = self._train if split == "train" else self._test
        return ds.complex

    def get_metadata(self):
        return {
            "task": "eigenvector_feedback", "config": self._cfg,
            "raw_shape": [self._nt, 4, 104], "nt": self._nt, "nr": 4, "nf": 104,
            "n_subbands": self._K, "subband_size": 8,
            "n_bs_per_map": self.n_bs_per_map, "n_ue_per_map": self.n_ue_per_map,
        }

    def env_index(self, sample_idx: int) -> int:
        return sample_idx // self.samples_per_env

    @property
    def n_samples(self, split: str = "test") -> int:
        return len(self._train) if split == "train" else len(self._test)


# ---------------------------------------------------------------------------
# 4. Run the demo
# ---------------------------------------------------------------------------

def main() -> int:
    import csieval
    from csieval import Evaluator, EvalConfig
    from csieval.core.registries import TaskRegistry

    # Register a fake eigenvector_feedback task that uses FakeEigenAdapter
    # We do this without touching the existing eigenvector_feedback module
    # by adding a new task under a different name (eigenvector_feedback_demo).
    @TaskRegistry.register("eigenvector_feedback_demo")
    class _DemoTask:
        name = "eigenvector_feedback_demo"
        input_layout = "paper"
        output_layout = "paper"
        primary_metric = "sgcs"
        def __init__(self):
            self._adapter = None
        def build_data(self, dataset_cfg, splits=("test",)):
            if self._adapter is None:
                self._adapter = FakeEigenAdapter(
                    n_train=dataset_cfg.get("n_train", 256),
                    n_test=dataset_cfg.get("n_test", 128),
                )
            return self._adapter
        def default_metrics(self):
            return None

    # Build & quick-train the tiny model
    print("[demo] Building tiny model ...")
    torch.manual_seed(0)
    net = TinyEVCsiNet(nt=32, n_subbands=13, embed_dim=32, num_layers=2, reduction=4, quant_bits=0)
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")
    print(f"  CR:     {net.get_compression_ratio():.4f}")

    # Quick MSE training so the model is not random
    print("[demo] Quick training (3 epochs, MSE) ...")
    train_ds = FakeEigenDataset(256, 32, 13, noise=0.05)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for epoch in range(3):
        for x in train_loader:
            opt.zero_grad()
            yhat = net(x)
            loss = nn.functional.mse_loss(yhat, x)
            loss.backward()
            opt.step()
        print(f"  epoch {epoch + 1}: loss = {loss.item():.4e}")

    # Save checkpoint
    ckpt_path = Path(tempfile.gettempdir()) / "ef_demo_tiny.pt"
    torch.save({"state_dict": net.state_dict(),
                "model_info": net.get_model_info()}, ckpt_path)
    print(f"[demo] Saved checkpoint: {ckpt_path}")

    # Build the EvalConfig
    out_dir = _PROJECT_ROOT / "results" / "ef_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(
        task="eigenvector_feedback_demo",
        checkpoint=str(ckpt_path),
        dataset={
            "n_train": 256, "n_test": 128,
            # override model kwargs via extra
            "model_class_path": "csieval.tests.demo.TinyEVCsiNet",
            "model_kwargs": {"nt": 32, "n_subbands": 13, "embed_dim": 32,
                              "num_layers": 2, "reduction": 4, "quant_bits": 0},
        },
        model={
            "name": "tiny",
            "task_name": "eigenvector_feedback",
        },
        model_class=TinyEVCsiNet,
        model_kwargs={"nt": 32, "n_subbands": 13, "embed_dim": 32,
                      "num_layers": 2, "reduction": 4, "quant_bits": 0},
        output_dir=str(out_dir),
        device="cpu",
        seed=0,
        splits=("test",),
        snr_levels_db=(5.0, 20.0, 40.0),  # small set for speed
        fewshot_samples=(16, 32),  # tiny for speed
        fewshot_epochs=2,
        quant_bits_sweep=(0, 4),
        latency_runs=5,
        report_formats=("json", "html", "markdown"),
    )

    print("[demo] Running Evaluator ...")
    ev = Evaluator(cfg)
    report = ev.run()
    report.print_summary()

    # Sanity-check report contents
    print(f"\n[demo] Report has {len(report.records)} records, "
          f"{len(report.skipped)} skipped.")
    for path in [
        out_dir / "metrics.json",
        out_dir / "report.html",
        out_dir / "report.md",
    ]:
        ok = path.exists()
        print(f"  [{'OK' if ok else 'MISSING'}] {path}")
        if ok and path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            print(f"      -> {len(data.get('metrics', []))} metrics, "
                  f"{len(data.get('skipped', []))} skipped")

    print("\nDEMO COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
