"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path

from ..core.report import EvalReport


CATEGORY_LABELS = {
    "task_performance": "1. Task Performance",
    "storage": "2. Deployment & Storage",
    "computation": "3. Computation Efficiency",
    "robustness": "4. Robustness & Generalization",
    "comparison": "Comparison",
}


def save(report: EvalReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    lines: list[str] = []
    lines.append("# CSI Feedback Evaluation Report")
    lines.append("")
    meta = report.meta
    lines.append(f"- **Task**: `{meta.get('task', '?')}`")
    lines.append(f"- **Device**: `{meta.get('device', '?')}`")
    if "checkpoint" in meta and meta["checkpoint"]:
        lines.append(f"- **Checkpoint**: `{meta['checkpoint']}`")
    lines.append(f"- **Timestamp**: `{meta.get('timestamp', '?')}`")
    if "model_name" in meta:
        lines.append(f"- **Model**: `{meta['model_name']}`")
    if "model_size_mb" in meta:
        lines.append(f"- **Size**: `{meta['model_size_mb']:.2f} MB`")
    lines.append("")

    by_cat: dict[str, list] = {}
    for r in report.records:
        by_cat.setdefault(r.category, []).append(r)

    # Render the first three categories as flat tables
    for cat_key in ("task_performance", "storage", "computation"):
        recs = by_cat.get(cat_key, [])
        if not recs:
            continue
        lines.append(f"## {CATEGORY_LABELS[cat_key]}")
        lines.append("")
        lines.append("| Metric | Value | Unit | Direction |")
        lines.append("|---|---|---|---|")
        for r in recs:
            arrow = "↑ higher better" if r.higher_is_better else "↓ lower better"
            val_str = _fmt(r.value)
            lines.append(f"| `{r.name}` | {val_str} | {r.unit} | {arrow} |")
        lines.append("")

    sub = report.sub_results or {}

    snr_curve = sub.get("robustness.snr_nmse.per_snr") or sub.get("snr_nmse.per_snr") or sub.get("snr_nmse_curve")
    quant_curve = sub.get("robustness.quant.per_quant_bits") or sub.get("quant.per_quant_bits") or sub.get("quantization_robustness_curve")
    ood = sub.get("ood")

    has_snr = bool(snr_curve and isinstance(snr_curve, list))
    has_quant = bool(quant_curve and isinstance(quant_curve, list) and bool(quant_curve))
    has_ood = bool(ood)

    if has_snr or has_quant or has_ood:
        lines.append("## 4. Robustness & Generalization")
        lines.append("")
        lines.append("> Per 3GPP TR 38.843 V19.0.0 Section 6.2, **Case 1** is the in-distribution "
                     "baseline (train &amp; test on the same scenario). "
                     "**Case 2** (cross-scenario zero-shot) and **Case 3** "
                     "(cross-scenario fine-tune) are evaluated per OOD target in "
                     "Section 4.3 below.")
        lines.append("")

    # 4.1 Noise robustness
    if has_snr:
        lines.append("### 4.1 Noise Robustness (SNR sweep)")
        lines.append("")
        lines.append("| SNR (dB) | NMSE (dB) | SGCS |")
        lines.append("|---|---|---|")
        for pt in snr_curve:
            snr = pt.get("snr_db", pt.get("snr", 0))
            if snr >= 9999:
                continue
            lines.append(f"| {snr:.1f} | {_fmt(pt.get('nmse_db'))} | {_fmt(pt.get('sgcs'))} |")
        lines.append("")

    # 4.2 Quantization robustness
    if has_quant:
        lines.append("### 4.2 Quantization Robustness (per bit-width)")
        lines.append("")
        lines.append("| Bit-width | NMSE (dB) | SGCS |")
        lines.append("|---|---|---|")
        for pt in quant_curve:
            lines.append(
                f"| {pt.get('quant_bits', '?')} | {_fmt(pt.get('nmse_db'))} | {_fmt(pt.get('sgcs'))} |"
            )
        lines.append("")

    # 4.3 Cross-Scenario Evaluation
    if has_ood:
        lines.append("### 4.3 Cross-Scenario Evaluation")
        lines.append("")
        lines.append("> **All Cross-Scenario Evaluation uses 2-bit quantization.** "
                     "Aggregation: per-map mean. Case 3 values shown are 10-shot results.")
        lines.append("")

        def _find_n10(per_n):
            if not per_n:
                return None
            for row in per_n:
                if row.get("n_support") == 10:
                    return row
            best, best_diff = None, float("inf")
            for row in per_n:
                d = abs(row.get("n_support", 0) - 10)
                if d < best_diff:
                    best_diff = d
                    best = row
            return best

        # Collect authoritative Case 1 baseline from OOD fine_tune results
        source_baseline = {}
        for tgt_sub in ood.values():
            fs = (tgt_sub.get("metrics") or {}).get("fine_tune") or {}
            c1_nmse = fs.get("case1_nmse_db")
            c1_sgcs = fs.get("case1_sgcs")
            if c1_nmse is not None:
                source_baseline["nmse_db"] = float(c1_nmse)
            if c1_sgcs is not None:
                source_baseline["sgcs"] = float(c1_sgcs)

        for tgt_name, tgt_sub in ood.items():
            tgt_meta = tgt_sub.get("target", {}) or {}
            desc = tgt_meta.get("description") or f"path={tgt_meta.get('path', '?')}"
            lines.append(f"#### {tgt_name}")
            lines.append(f"_{desc}_")
            lines.append("")
            if "error" in tgt_sub:
                lines.append(f"> **Error:** {tgt_sub['error']}")
                lines.append("")
                continue
            m = tgt_sub.get("metrics", {}) or {}

            c1 = m.get("id_baseline") or {}
            c2 = m.get("zero_shot") or {}
            c3 = m.get("fine_tune") or {}
            per_n = c3.get("per_n", []) or []

            sb_nmse = source_baseline.get("nmse_db") or c1.get("nmse_db")
            sb_sgcs = source_baseline.get("sgcs") or c1.get("sgcs")
            c2_nmse = c2.get("nmse_db")
            c2_sgcs = c2.get("sgcs")
            c2_std_nmse = c2.get("std_nmse_db")
            c2_std_sgcs = c2.get("std_sgcs")
            c1_std_nmse = c1.get("std_nmse_db")
            c1_std_sgcs = c1.get("std_sgcs")

            n10 = _find_n10(per_n)
            if n10:
                c3_nmse = n10.get("nmse_db")
                c3_sgcs = n10.get("sgcs")
                c3_n = n10.get("n_support", 10)
                c3_tag = f" ({c3_n}-shot)"
            else:
                c3_nmse = c3.get("best_nmse_db")
                c3_sgcs = c3.get("best_sgcs")
                c3_tag = ""

            gap_nmse_c2 = (c2_nmse - sb_nmse) if (c2_nmse is not None and sb_nmse is not None) else None
            gap_nmse_c3 = (c3_nmse - sb_nmse) if (c3_nmse is not None and sb_nmse is not None) else None
            gdr_c2 = ((c2_sgcs - sb_sgcs) / sb_sgcs) if (c2_sgcs is not None and sb_sgcs is not None and sb_sgcs != 0) else None
            gdr_c3 = ((c3_sgcs - sb_sgcs) / sb_sgcs) if (c3_sgcs is not None and sb_sgcs is not None and sb_sgcs != 0) else None

            lines.append("| Metric | Baseline | Cross-scenario Zero-Shot | Cross-scenario Fine-Tune |")
            lines.append("|---|---|---|---|")
            lines.append(f"| NMSE — Per-Map Mean (dB) | `{_opt(sb_nmse)}` | `{_opt(c2_nmse)}` | `{_opt(c3_nmse)}`{c3_tag} |")
            lines.append(f"| SGCS — Per-Map Mean | `{_opt(sb_sgcs)}` | `{_opt(c2_sgcs)}` | `{_opt(c3_sgcs)}`{c3_tag} |")
            lines.append(f"| Per-Map NMSE — Std (dB) | `{_opt(c1_std_nmse)}` | `{_opt(c2_std_nmse)}` | `—` |")
            lines.append(f"| Per-Map SGCS — Std | `{_opt(c1_std_sgcs)}` | `{_opt(c2_std_sgcs)}` | `—` |")
            lines.append(f"| Gap NMSE (dB) | `—` | `{_opt(gap_nmse_c2)}` | `{_opt(gap_nmse_c3)}` |")
            lines.append(f"| SGCS Generalized Decay Rate | `—` | `{_opt(gdr_c2)}` | `{_opt(gdr_c3)}` |")
            lines.append("")

            if per_n:
                lines.append(f"**Few-Shot Fine-Tuning Curve (n_support=[0,5,10,20,50,100,300])**")
                lines.append("")
                lines.append("| n_support | NMSE (dB) | SGCS |")
                lines.append("|---|---|---|")
                for r in per_n:
                    n = r.get("n_support", "?")
                    tag = " _(Zero-Shot)_" if r.get("is_zeroshot") else ""
                    nmse = _opt(r.get("nmse_db"))
                    sgcs_v = r.get("sgcs")
                    sgcs_str = _opt(sgcs_v) if sgcs_v is not None and sgcs_v == sgcs_v else "—"
                    lines.append(f"| {n}{tag} | {nmse} | {sgcs_str} |")
                lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _opt(v) -> str:
    """Format a value for the OOD MD table: float with 4dp, None → em-dash."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(val)}" for k, val in list(v.items())[:5])
    if isinstance(v, list):
        if len(v) > 5:
            return f"[{', '.join(_fmt(x) for x in v[:3])}…({len(v)} items)]"
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)
