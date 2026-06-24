"""Efficiency / deployment metrics.

Covers:
  - model_size_mb       : on-disk size of state_dict
  - params_m            : #parameters in millions
  - peak_memory_mb      : peak memory during inference (CPU + GPU)
  - quant_bits          : quantization bit width of the model
  - compression_ratio   : compressed_dim / total_dim
  - load_time_s         : checkpoint load wall time
  - quant_feedback_overhead_bytes : bytes transmitted over-the-air for
                                    one feedback codeword (CR × quant_bits)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict

import torch

from ..core.context import EvalContext
from ..core.registries import MetricRegistry


# ---------------------------------------------------------------------------
# ModelSize
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class ModelSize:
    """On-disk state_dict size in MB (after dumping)."""

    name = "size"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "MB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        size_mb = ctx.model.get_state_dict_mb()
        return {"value": float(size_mb), "unit": "MB"}


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class ParamsM:
    """Total #parameters in millions."""

    name = "params"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "M"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        params = sum(p.numel() for p in ctx.model.parameters() if hasattr(p, "numel"))
        # Duck-typed: if the model has .parameters() we count; otherwise 0
        return {"value": params / 1e6, "raw": int(params), "unit": "M"}


# ---------------------------------------------------------------------------
# PeakMemory
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class PeakMemory:
    """Peak memory during one forward pass (MB).

    CPU uses tracemalloc; CUDA uses torch.cuda.max_memory_allocated().
    """

    name = "memory"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "MB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        # Warm up CUDA allocator
        ctx.model.eval()
        x = torch.randn((1, *ctx.model.get_input_shape()), device=ctx.device)
        with torch.no_grad():
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(ctx.device) / (1024 ** 2)
        else:
            import tracemalloc
            tracemalloc.start()
            with torch.no_grad():
                _ = ctx.model.forward(x)
            current, peak_py = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = max(current, peak_py) / (1024 ** 2)
        return {"value": float(peak), "unit": "MB"}


# ---------------------------------------------------------------------------
# QuantBits
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset(["model.quant_bits"]))
class QuantBits:
    """Quantization bit width (0 means no quantization)."""

    name = "bitwidth"
    category = "storage"
    higher_is_better = False
    requires = frozenset(["model.quant_bits"])

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        q = int(ctx.model.get_quant_bits())
        return {"value": q, "effective": q, "unit": "bits"}


# ---------------------------------------------------------------------------
# CompressionRatio
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset(["model.compression_ratio"]))
class CompressionRatio:
    """Feedback code dimension / total dimension."""

    name = "compression"
    category = "storage"
    higher_is_better = False
    requires = frozenset(["model.compression_ratio"])
    unit = ""

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        cr = float(ctx.model.get_compression_ratio())
        info = {}
        if ctx.has("model.info"):
            try:
                info = ctx.model.get_model_info()
            except Exception:
                info = {}
        return {
            "value": cr,
            "compressed_dim": info.get("compressed_dim"),
            "total_dim": info.get("total_dim"),
            "reduction": info.get("reduction"),
        }


# ---------------------------------------------------------------------------
# CSIReductionRate
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset(["model.compression_ratio"]))
class CSIReductionRate:
    """CSI Feedback overhead reduction rate vs Rel-16/17 Type II codebook baseline.

    Rel-16/17 Type II codebook feedback overhead:
      - Type II (2x2):  4 bits per real entry in the (Nt×Nr) beam matrix
        ≈ 4 * Nt * Nr * (1 + 2*log2(Nt)) bits per subband
      - For 2.6 GHz (Nt=32, Nr=4): 4 * 32 * 4 * (1 + 2*log2(32))
        = 512 * (1 + 10) = 5632 bits ≈ 704 bytes per subband (≈8800 bits total)
      - Per-subband overhead for Type II ≈ 4 * Nt * Nr = 512 bits

    Our model's overhead: compressed_dim * quant_bits / 8 bytes per codeword.
    Reduction rate = (baseline_overhead - model_overhead) / baseline_overhead * 100%

    Higher is better (positive = we reduce overhead vs Type II).
    """

    name = "csi_reduction_rate"
    category = "storage"
    higher_is_better = True
    requires = frozenset(["model.compression_ratio"])
    unit = "%"

    # Type II codebook overhead per subband (in bits), keyed by (Nt, Nr)
    TYPEII_OVERHEAD_BITS: Dict[tuple, float] = {
        # (Nt, Nr): bits per subband
        (32, 4): 512.0,   # 2.6 GHz: 4 * Nt * Nr
        (64, 4): 1024.0,  # 3.5 GHz
        (256, 8): 8192.0, # 7 GHz: 4 * 256 * 8
    }

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if ctx.has("model.info"):
            try:
                info = ctx.model.get_model_info() or {}
            except Exception:
                info = {}
        compressed_dim = info.get("compressed_dim")
        if compressed_dim is None:
            cr = float(ctx.model.get_compression_ratio())
            total_dim = info.get("total_dim", 0) or 0
            compressed_dim = int(round(cr * total_dim)) if total_dim else 0
        q = int(ctx.model.get_quant_bits()) if ctx.has("model.quant_bits") else 0
        bits_per_code = max(q, 4)
        model_bits = int(compressed_dim) * bits_per_code
        model_overhead_bits = float(model_bits)

        # Determine Nt, Nr from model info or config
        nt = info.get("nt") or 32
        nr = info.get("nr") or 4
        baseline_bits = self.TYPEII_OVERHEAD_BITS.get((nt, nr), 512.0)
        # Scale by number of subbands if available
        n_subbands = info.get("n_subbands")
        if n_subbands:
            baseline_bits *= n_subbands
            model_overhead_bits *= n_subbands

        reduction_pct = 0.0
        if baseline_bits > 0:
            reduction_pct = float(100.0 * (baseline_bits - model_overhead_bits) / baseline_bits)

        return {
            "value": reduction_pct,
            "unit": "%",
            "baseline_bits": baseline_bits,
            "model_bits": model_overhead_bits,
            "compressed_dim": int(compressed_dim),
            "quant_bits": int(bits_per_code),
        }


# ---------------------------------------------------------------------------
# LoadTime
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class LoadTime:
    """Time to load state_dict into the model (seconds)."""

    name = "loadtime"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "s"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ckpt_path = ctx.config.checkpoint
        if not ckpt_path or not os.path.exists(ckpt_path):
            return {"value": None, "note": "no checkpoint path"}
        t0 = time.perf_counter()
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        # Move model to CPU temporarily for fair timing
        original_device = next(ctx.model.parameters()).device
        ctx.model.cpu()
        try:
            ctx.model.load_state_dict(sd, strict=False)
        except Exception as e:
            return {"value": None, "note": f"load failed: {e}"}
        finally:
            ctx.model.to(original_device)
        elapsed = time.perf_counter() - t0
        return {"value": float(elapsed), "unit": "s"}


# ---------------------------------------------------------------------------
# QuantFeedbackOverhead
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset(["model.compression_ratio"]))
class QuantFeedbackOverhead:
    """Over-the-air feedback overhead in bytes per codeword.

    Definition: compressed_dim * quant_bits / 8 bytes.
    For an unquantized model (quant_bits=0) we use 4 bytes (float32) per
    code element by convention.

    Lower is better.
    """

    name = "overhead"
    category = "storage"
    higher_is_better = False
    requires = frozenset(["model.compression_ratio"])
    unit = "bytes"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if ctx.has("model.info"):
            try:
                info = ctx.model.get_model_info() or {}
            except Exception:
                info = {}
        compressed_dim = info.get("compressed_dim")
        if compressed_dim is None:
            # fall back: try CR * total_dim
            cr = float(ctx.model.get_compression_ratio())
            total_dim = info.get("total_dim", 0) or 0
            compressed_dim = int(round(cr * total_dim)) if total_dim else 0
        q = int(ctx.model.get_quant_bits()) if ctx.has("model.quant_bits") else 0
        bits_per_code = max(q, 4)  # unquantized -> 4 bytes (float32) convention
        bytes_per_codeword = int(compressed_dim) * bits_per_code // 8
        return {
            "value": bytes_per_codeword,
            "compressed_dim": int(compressed_dim),
            "quant_bits_effective": int(bits_per_code),
            "unit": "bytes/codeword",
        }
