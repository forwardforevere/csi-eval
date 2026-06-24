"""Computation metrics: latency, FLOPs, MACs."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from ..core.context import EvalContext
from ..core.registries import MetricRegistry


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset())
class Latency:
    """Average inference latency in ms (over ``latency_runs`` runs)."""

    name = "latency"
    category = "computation"
    higher_is_better = False
    requires = frozenset()
    unit = "ms"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        runs = int(ctx.config.latency_runs)
        input_shape = ctx.model.get_input_shape()

        results: Dict[str, float] = {}
        for bs in (1, max(1, min(16, runs))):
            results[f"bs{bs}"] = self._measure(ctx, input_shape, batch_size=bs, runs=runs)

        # Primary value: batch_size=1
        return {"value": results["bs1"], "by_batch_size": results, "unit": "ms"}

    @torch.no_grad()
    def _measure(self, ctx: EvalContext, input_shape, batch_size: int, runs: int) -> float:
        ctx.model.eval().to(ctx.device)
        x = torch.randn((batch_size, *input_shape), device=ctx.device)
        # Warmup
        for _ in range(max(1, runs // 10)):
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / runs


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset())
class FLOPs:
    """FLOPs (floating point operations) per inference.

    Strategy:
      1. If the model exposes estimate_macs, use 2*macs.
      2. Otherwise, try thop.profile (if installed).
      3. Otherwise, return None.
    """

    name = "flops"
    category = "computation"
    higher_is_better = False
    requires = frozenset()
    unit = "FLOPs"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        macs = None
        if ctx.has("model.macs"):
            try:
                macs = int(ctx.model.estimate_macs(batch_size=1))
            except Exception:
                macs = None
        if macs is None:
            try:
                import thop  # type: ignore
                x = torch.randn((1, *ctx.model.get_input_shape()), device=ctx.device)
                macs, _ = thop.profile(ctx.model, inputs=(x,), verbose=False)
            except Exception:
                macs = None
        if macs is None:
            return {"value": None, "note": "FLOPs profiler not available"}
        return {"value": int(2 * macs), "macs": int(macs), "unit": "FLOPs"}


# ---------------------------------------------------------------------------
# MAC
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset(["model.macs"]))
class MAC:
    """Multiply-accumulate operations per inference."""

    name = "macs"
    category = "computation"
    higher_is_better = False
    requires = frozenset(["model.macs"])
    unit = "MACs"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        try:
            macs = int(ctx.model.estimate_macs(batch_size=1))
        except Exception as e:
            return {"value": None, "note": f"estimate_macs failed: {e}"}
        return {"value": macs, "unit": "MACs"}
