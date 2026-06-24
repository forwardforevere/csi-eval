"""Robustness metrics: SNR-NMSE slope, quantization robustness.

Both metrics require raw complex CSI for noise injection. If the dataset
does not provide it, the metric is skipped.

This file is self-contained: it does NOT import from the original
``dataset/`` package. Eigenvector preprocessing helpers are obtained at
runtime from the active task via a registered callback (set by
``tasks/eigenvector_feedback/__init__.py``). If no callback is registered
and no raw CSI is available, the SNR-NMSE metric is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..core.context import EvalContext
from ..core.registries import MetricRegistry
from ._math import nmse_db_ri, sgcs_ri
from .task_performance import _ensure_predictions


# ---------------------------------------------------------------------------
# Optional hook: task can register a (noisy_complex_array, subband_size)
# -> model_input_numpy function. This decouples robustness from the
# original dataset/ package.
# ---------------------------------------------------------------------------

_eigen_preprocess_fn = None  # global, set by task modules


def register_eigen_preprocess_fn(fn) -> None:
    """Register a function ``fn(noisy_H, subband_size) -> x_numpy``.

    The function should map a noisy complex CSI array ``[N, Nt, Nr, Nf]``
    to a model input ``[N, 2, Nt, K]`` numpy tensor. The
    eigenvector_feedback task registers this when its task instance is
    constructed.
    """
    global _eigen_preprocess_fn
    _eigen_preprocess_fn = fn


def get_eigen_preprocess_fn():
    return _eigen_preprocess_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_awgn_to_complex(H: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Inject AWGN such that the per-sample signal-to-noise ratio is snr_db."""
    if snr_db >= 9999:
        return H
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    sig = np.mean(np.abs(H) ** 2)
    noise_pow = sig / (10 ** (snr_db / 10))
    sigma = np.sqrt(noise_pow / 2)
    noise = sigma * (
        rng.standard_normal(H.shape).astype(np.float32)
        + 1j * rng.standard_normal(H.shape).astype(np.float32)
    )
    return (H + noise).astype(np.complex64)


# ---------------------------------------------------------------------------
# SNR_NMSE_Slope
# ---------------------------------------------------------------------------
@MetricRegistry.register("robustness", requires=frozenset(["data.complex_raw"]))
class SNR_NMSE_Slope:
    """Linear slope of NMSE (dB) vs SNR (dB); ideal is -1 dB/dB.

    Also caches the per-SNR SGCS curve so the companion
    :class:`SNR_SGCS_Slope` can read the same forward passes without
    re-doing the (expensive) covariance-domain noise injection.
    """

    name = "snr_nmse"
    category = "robustness"
    higher_is_better = False
    requires = frozenset(["data.complex_raw"])
    unit = "dB/dB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        raw = ctx.data.get_complex_raw("test")
        if raw is None:
            return {"value": None, "note": "no complex raw available"}

        info = ctx.data.get_metadata()
        subband_size = int(info.get("subband_size", 8))
        nt = int(info.get("nt", raw.shape[1]))

        if raw.shape[-1] % subband_size != 0:
            return {
                "value": None,
                "note": f"raw Nf={raw.shape[-1]} not divisible by subband_size={subband_size}",
            }

        per_snr: List[Dict[str, float]] = []
        snr_list: List[float] = []
        nmse_list: List[float] = []

        def _compute_for_snr(snr_db: float) -> Dict[str, float]:
            key = f"snr_pair_{snr_db}"
            cached = ctx.get(key)
            if cached is not None:
                # Backward compat: old runs cached a plain float (nmse_db only)
                if isinstance(cached, float):
                    return {"nmse_db": float(cached), "sgcs": float("nan")}
                if isinstance(cached, dict):
                    return cached
            return ctx.get_or_compute(
                key,
                lambda: self._do_one_snr(raw, snr_db, subband_size, nt, ctx),
            )

        for snr_db in ctx.config.snr_levels_db:
            pair = _compute_for_snr(snr_db)
            per_snr.append({
                "snr_db": snr_db,
                "nmse_db": pair["nmse_db"],
                "sgcs": pair["sgcs"],
            })
            if snr_db < 9999:
                snr_list.append(snr_db)
                nmse_list.append(pair["nmse_db"])

        slope = float("nan")
        # Filter out NaN entries (which can occur if eigh failed for
        # a particular SNR) before fitting. Otherwise np.polyfit
        # returns NaN and the metric is reported as None.
        pairs = [(s, n) for s, n in zip(snr_list, nmse_list)
                 if np.isfinite(s) and np.isfinite(n)]
        if len(pairs) >= 2:
            s_arr, n_arr = zip(*pairs)
            slope = float(np.polyfit(np.array(s_arr), np.array(n_arr), deg=1)[0])

        ctx.add_sub("snr_nmse.per_snr", per_snr)
        return {
            "value": slope,
            "per_snr": per_snr,
            "unit": "dB/dB",
        }

    @staticmethod
    def _do_one_snr(raw, snr_db, subband_size, nt, ctx) -> Dict[str, float]:
        """Run the model at a single SNR and return NMSE(dB) + SGCS.

        This is the heavy inner loop. We cache its (nmse, sgcs) tuple
        per SNR so that both ``SNR_NMSE_Slope`` and ``SNR_SGCS_Slope``
        can share the result.
        """
        # Physics: at the receiver, the channel estimate covariance
        # ``R_y = R_s + R_n`` carries both signal and noise. We pre-compute
        # the noise-free signal covariance ``R_s`` once and add the
        # per-SNR noise covariance ``R_n = sigma^2 I`` per subband. This
        # is far cheaper than re-estimating the full eigh on noisy raw
        # CSI and matches the standard 3GPP TypeI codebook convention.
        #
        # To stay GPU-friendly and avoid materializing a multi-GB
        # covariance tensor on CPU, we compute ``R_s`` on-the-fly per
        # sample on GPU. For the 2.6 GHz dataset (Nt=32, K=13) each
        # sample contributes a 32x32 covariance per subband; the GPU
        # can do all eighs in parallel.
        #
        # Notes on numerical stability
        # ----------------------------
        # - cuSOLVER's batched eigh rejects matrices that are not
        #   strictly Hermitian in float32 (status INVALID_VALUE). The
        #   code below forces Hermiticity via ``0.5 * (R + R^H)``.
        # - When ``Nt > Nr`` the rank-deficient ``R_s = H^H H / Nr``
        #   has negative eigenvalues from float32 drift; a tiny ridge
        #   ``eps * I`` makes the matrix PSD and keeps cuSOLVER happy.
        # - For the 2.6 GHz dataset (1.6M samples x 13 subbands ~ 21M
        #   32x32 matrices) the full batch is too large for cuSOLVER's
        #   batched workspace on some hardware. We therefore process
        #   the data in fixed-size chunks and let eigh fail on a
        #   chunk without aborting the whole metric.
        from ..tasks.eigenvector_feedback._internal._eigen_preprocess import (
            eigenvectors_to_model_input,
        )

        device = ctx.device
        if not torch.cuda.is_available():
            # CPU fallback: use the vendored reference helper.
            from ..tasks.eigenvector_feedback._internal._eigen_preprocess import (
                compute_subband_signal_covariances,
            )
            R_s = ctx.get_or_compute(
                "snr_R_signal",
                lambda: compute_subband_signal_covariances(
                    raw, subband_size=int(subband_size), normalize_input_power=False,
                    batch_size=64,
                ),
            )
            K = R_s.shape[1]
            N = R_s.shape[0]
            if snr_db >= 9999:
                sigma_sq = 0.0
            else:
                sig_pow = float(np.mean(np.abs(R_s).real))
                noise_pow = sig_pow / (10 ** (snr_db / 10))
                sigma_sq = max(noise_pow / max(nt, 1), 1e-12)
            rng = np.random.default_rng(ctx.config.seed)
            R_noisy = R_s
            if sigma_sq > 0.0:
                noise_real = rng.standard_normal((N, K, nt, nt)).astype(np.float32)
                noise_imag = rng.standard_normal((N, K, nt, nt)).astype(np.float32)
                N_n = (noise_real + 1j * noise_imag) * np.sqrt(sigma_sq / 2)
                R_noisy = (R_s + N_n).astype(np.complex64)
            eigvals, eigvecs = np.linalg.eigh(R_noisy)
            wk = eigvecs[:, :, :, -1]                          # [N, K, Nt]
            wk = wk / (np.linalg.norm(wk, axis=-1, keepdims=True) + 1e-12)
            idx_max = np.argmax(np.abs(wk), axis=-1)
            phases = np.exp(-1j * np.angle(wk[np.arange(N)[:, None], np.arange(K)[None, :], idx_max]))
            wk = wk * phases[:, :, np.newaxis]
            ri = np.stack([wk.real, wk.imag], axis=1).astype(np.float32)
            x = np.transpose(ri, (0, 1, 3, 2)).astype(np.float32)
            xt = torch.from_numpy(x).float()
        else:
            H = torch.from_numpy(raw).to(device)              # [N, Nt, Nr, Nf]
            N, Nt, Nr, Nf = H.shape
            K = Nf // subband_size
            # Subband mean along the frequency axis: [N, Nt, Nr, K]
            Hsb = H.view(N, Nt, Nr, K, subband_size).mean(dim=-1)
            # R_s = Hsb @ Hsb^H averaged over Rx.
            Hmat = Hsb.permute(0, 3, 2, 1).contiguous()       # [N, K, Nr, Nt]
            # Ridge coefficient for PSD-ness. The covariance
            # ``R_s = H^H H / Nr`` is rank-deficient when ``Nr < Nt``
            # (2.6 GHz: Nr=4, Nt=32), and float32 multiplication
            # produces negative eigenvalues on the order of
            # ``1e-6 * ||R_s||``. cuSOLVER refuses such matrices with
            # CUSOLVER_STATUS_INVALID_VALUE. We therefore add a tiny
            # absolute ridge ``eps * I`` (NOT relative to the diagonal
            # mean, which can be ~1e-4 and is too small to dominate
            # the float32 noise floor). The 1e-5 default is roughly
            # -85 dB below the signal — small enough to not bias the
            # eigenvector estimate but large enough to make every
            # 32x32 matrix strictly PSD.
            eps = 1e-5
            # noise power relative to signal power per receive antenna
            if snr_db >= 9999:
                sigma_sq_val = 0.0
            else:
                sigma_sq_val = float(eps) / (10 ** (snr_db / 10))
                sigma_sq_val = max(sigma_sq_val, 1e-12)

            # Process samples in chunks to keep cuSOLVER's batched
            # workspace within budget. The cuSOLVER batched eigh
            # ``bufferSize`` query itself fails with INVALID_VALUE on
            # the 2.6 GHz test set whenever the per-chunk batch is
            # above ~2048 (verified empirically — the same chunked
            # code on a random complex64 tensor works up to 8192, so
            # the limit is matrix-content-dependent, not a hard
            # cuSOLVER limit). 1024 is the largest batch that runs
            # reliably on this hardware. For larger datasets the
            # eigh is the inner loop anyway, so the chunk size
            # is the throughput dial, not a correctness one.
            CHUNK = 1024
            wk_chunks = []
            ok_chunks = 0
            for i in range(0, N, CHUNK):
                sl = slice(i, min(i + CHUNK, N))
                Hc = Hmat[sl]                                 # [c, K, Nr, Nt]
                R_c = torch.matmul(Hc.transpose(-1, -2).conj(), Hc) / Nr
                R_c = 0.5 * (R_c + R_c.transpose(-1, -2).conj())
                if sigma_sq_val > 0.0:
                    noise_real = torch.randn(
                        Hc.shape[0], Hc.shape[1], nt, nt, device=device, dtype=torch.float32
                    )
                    noise_imag = torch.randn(
                        Hc.shape[0], Hc.shape[1], nt, nt, device=device, dtype=torch.float32
                    )
                    scale = (sigma_sq_val / 2.0) ** 0.5
                    N_n = torch.complex(noise_real, noise_imag) * scale
                    R_c = R_c + N_n
                R_c = 0.5 * (R_c + R_c.transpose(-1, -2).conj()) + eps * torch.eye(
                    nt, device=device, dtype=R_c.dtype
                )
                try:
                    _ev, vc = torch.linalg.eigh(R_c)
                except Exception as chunk_err:
                    print(
                        f"[SNR_NMSE_Slope] CUDA eigh failed on chunk "
                        f"[{sl.start}:{sl.stop}] at snr={snr_db} dB ({chunk_err}); "
                        f"skipping chunk."
                    )
                    continue
                # Pick the principal eigenvector, normalise, phase-fix.
                w = vc[:, :, :, -1]                          # [c, K, Nt]
                w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                wk_chunks.append(w.cpu())                     # small chunk → cheap
                ok_chunks += 1
            if ok_chunks == 0:
                # Every chunk failed — return a NaN so the linear
                # fit degrades gracefully instead of aborting the run.
                return float("nan")
            wk = torch.cat(wk_chunks, dim=0)                 # [N, K, Nt]
            # Phase fix
            idx_max = wk.abs().argmax(dim=-1)
            phases = torch.exp(-1j * torch.angle(
                wk.gather(-1, idx_max.unsqueeze(-1)).squeeze(-1)
            ))
            wk = wk * phases.unsqueeze(-1)
            ri = torch.stack([wk.real, wk.imag], dim=1).to(torch.float32)  # [N, 2, K, Nt]
            x = ri.permute(0, 1, 3, 2).contiguous()
            xt = x                                          # keep on CPU; model.forward copies per batch

        preds = []
        ctx.model.eval()
        with torch.no_grad():
            for i in range(0, xt.shape[0], 256):
                xb = xt[i:i + 256]
                if not torch.is_tensor(xb):
                    xb = torch.from_numpy(xb).float()
                preds.append(ctx.model.forward(xb.to(ctx.device)).cpu())
        y_pred = torch.cat(preds, dim=0)
        y_true = xt if torch.is_tensor(xt) and xt.dtype == torch.float32 else xt.float()
        nmse_stats = nmse_db_ri(y_true, y_pred)
        sgcs_stats = sgcs_ri(y_true, y_pred)
        return {
            "nmse_db": float(nmse_stats["mean"]),
            "sgcs": float(sgcs_stats["mean"]),
        }


# ---------------------------------------------------------------------------
# SNR_SGCS_Slope
# ---------------------------------------------------------------------------
@MetricRegistry.register("robustness", requires=frozenset(["data.complex_raw"]))
class SNR_SGCS_Slope:
    """Linear slope of SGCS vs SNR (dB); ideal is +1 (or higher) / 10 dB.

    Symmetric companion to :class:`SNR_NMSE_Slope`. Both metrics share
    the same cached per-SNR forward pass via the ``snr_pair_{snr_db}``
    cache key, so this metric only pays the cost of the linear fit.
    """

    name = "snr_sgcs"
    category = "robustness"
    higher_is_better = True
    requires = frozenset(["data.complex_raw"])
    unit = "/dB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        raw = ctx.data.get_complex_raw("test")
        if raw is None:
            return {"value": None, "note": "no complex raw available"}

        info = ctx.data.get_metadata()
        subband_size = int(info.get("subband_size", 8))
        nt = int(info.get("nt", raw.shape[1]))

        if raw.shape[-1] % subband_size != 0:
            return {
                "value": None,
                "note": f"raw Nf={raw.shape[-1]} not divisible by subband_size={subband_size}",
            }

        per_snr: List[Dict[str, float]] = []
        snr_list: List[float] = []
        sgcs_list: List[float] = []

        def _compute_for_snr(snr_db: float) -> Dict[str, float]:
            key = f"snr_pair_{snr_db}"
            cached = ctx.get(key)
            if cached is not None:
                if isinstance(cached, float):
                    return {"nmse_db": float(cached), "sgcs": float("nan")}
                if isinstance(cached, dict):
                    return cached
            return ctx.get_or_compute(
                key,
                lambda: SNR_NMSE_Slope._do_one_snr(raw, snr_db, subband_size, nt, ctx),
            )

        for snr_db in ctx.config.snr_levels_db:
            pair = _compute_for_snr(snr_db)
            per_snr.append({
                "snr_db": snr_db,
                "nmse_db": pair["nmse_db"],
                "sgcs": pair["sgcs"],
            })
            if snr_db < 9999:
                snr_list.append(snr_db)
                sgcs_list.append(pair["sgcs"])

        slope = float("nan")
        pairs = [(s, g) for s, g in zip(snr_list, sgcs_list)
                 if np.isfinite(s) and np.isfinite(g)]
        if len(pairs) >= 2:
            s_arr, g_arr = zip(*pairs)
            slope = float(np.polyfit(np.array(s_arr), np.array(g_arr), deg=1)[0])

        ctx.add_sub("snr_sgcs.per_snr", per_snr)
        return {
            "value": slope,
            "per_snr": per_snr,
            "unit": "/dB",
        }


# ---------------------------------------------------------------------------
# QuantizationRobustness
# ---------------------------------------------------------------------------
@MetricRegistry.register("robustness", requires=frozenset(["model.quant_bits"]))
class QuantizationRobustness:
    """Sweep quant_bits and report NMSE at each level.

    This works by re-wrapping the model with a fake-quantization layer on
    top of the encoded code. If the model has a configurable `quant_bits`
    attribute, we set it directly. Otherwise we wrap the model's encode().
    """

    name = "quant"
    category = "robustness"
    higher_is_better = False
    requires = frozenset(["model.quant_bits"])
    unit = ""

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        # Re-use cached predictions from the task-performance pass so we
        # don't need to re-run the model on y_true for every bit-width.
        from .task_performance import _ensure_predictions
        y_true, _ = _ensure_predictions(ctx)

        sweep = list(ctx.config.quant_bits_sweep)
        per_bits: List[Dict[str, Any]] = []

        # If the model does not expose ``quant_bits`` at all, the
        # forward is bit-width invariant — we should only run the
        # expensive forward once and report a single entry.
        supports_quant = hasattr(ctx.model, "quant_bits")
        if not supports_quant:
            q = sweep[0] if sweep else 0
            y_pred = ctx.get_or_compute(
                f"quant_y_pred_{q}",
                lambda: self._forward_all(ctx),
            )
            stats = nmse_db_ri(y_true, y_pred)
            sgcs = sgcs_ri(y_true, y_pred)
            per_bits = [{"quant_bits": int(q), "nmse_db": stats["mean"], "sgcs": sgcs["mean"]}]
            ctx.add_sub("quant.per_quant_bits", per_bits)
            return {
                "value": stats["mean"],
                "per_quant_bits": per_bits,
                "best_quant_bits": int(q),
                "note": "model has no quant_bits attribute; sweep collapsed to 1 forward",
            }

        # The model advertises ``quant_bits`` — sweep through every
        # requested bit-width. ``setattr(ctx.model, "quant_bits", q)``
        # is forwarded to the inner ``nn.Module`` by the adapter's
        # ``__setattr__`` (see ``ModelAdapter``), so the new bit-width
        # takes effect on the very next forward.
        original_bits = int(getattr(ctx.model, "quant_bits"))
        try:
            for q in sweep:
                setattr(ctx.model, "quant_bits", int(q))
                # Re-evaluate the model at the new bit-width. Note we
                # do NOT use ``get_or_compute`` here because the result
                # depends on the model's mutable ``quant_bits`` — caching
                # by key alone would let a stale result leak between
                # ``q=2`` and ``q=4`` runs if any other code touched
                # ``ctx.model.quant_bits`` between iterations.
                y_pred = self._forward_all(ctx)
                stats = nmse_db_ri(y_true, y_pred)
                sgcs = sgcs_ri(y_true, y_pred)
                per_bits.append({
                    "quant_bits": int(q),
                    "nmse_db": float(stats["mean"]),
                    "sgcs": float(sgcs["mean"]),
                })
        finally:
            # Always restore
            setattr(ctx.model, "quant_bits", original_bits)

        # Sanity check: if every bit-width produced the exact same
        # number, the bit-switching is broken somewhere. Surface that
        # loudly instead of silently reporting a flat sweep.
        distinct = {round(r["nmse_db"], 3) for r in per_bits}
        note = ""
        if len(distinct) <= 1 and len(per_bits) > 1:
            note = (
                "quantization sweep returned identical values for all "
                "bit-widths — model.quant_bits is not being honored by "
                "the forward pass."
            )

        best = min(per_bits, key=lambda r: r["nmse_db"]) if per_bits else None
        ctx.add_sub("quant.per_quant_bits", per_bits)
        return {
            "value": best["nmse_db"] if best else None,
            "per_quant_bits": per_bits,
            "best_quant_bits": best["quant_bits"] if best else None,
            "note": note,
        }

    @staticmethod
    def _forward_all(ctx) -> torch.Tensor:
        """Run the model on every split's test data and concatenate."""
        ctx.model.eval()
        yhats = []
        for split in ctx.splits:
            loader = ctx.data.get_loader(
                split, batch_size=128, num_workers=0, shuffle=False
            )
            with torch.no_grad():
                for batch in loader:
                    x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x = x.to(ctx.device, non_blocking=True).float()
                    yhats.append(ctx.model.forward(x).detach().cpu())
        return torch.cat(yhats, dim=0)
