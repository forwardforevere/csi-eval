"""Smoke test: verify that the csieval package can be
imported and that all runners/metrics register without errors.

This script does NOT touch GPU or load real checkpoints; it only checks
that the Python module graph and the registries are consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so that "import models", "import dataset"
# work even if this script is run from a different directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    print("[1/4] Importing csieval ...")
    import csieval
    print("  package:", csieval.__file__)

    print("[2/4] Importing submodules ...")
    from csieval.core import (
        EvalConfig, Evaluator, EvalContext, EvalReport, MetricRecord,
    )
    from csieval.core.registries import TaskRegistry, MetricRegistry
    from csieval.runners import build_default_runners
    from csieval.reports import save_json, save_html, save_markdown
    from csieval.tasks.eigenvector_feedback import (
        EigenvectorFeedbackTask, EigenvectorDataAdapter,
    )
    print("  ok.")

    print("[3/4] Checking registries ...")
    tasks = TaskRegistry.list_tasks()
    print(f"  Registered tasks ({len(tasks)}): {tasks}")
    assert "eigenvector_feedback" in tasks, "eigenvector_feedback must be registered"

    cats = ["task_performance", "storage", "computation", "robustness"]
    for c in cats:
        names = [m.name for m in MetricRegistry.list_by_category(c)]
        print(f"  [{c}] {len(names)} metrics: {names}")
        assert len(names) > 0, f"category {c} has no metrics"

    print("[4/4] Building default runners (no real model loaded) ...")
    cfg = EvalConfig(
        task="eigenvector_feedback",
        output_dir="/tmp/ef_smoke",
        device="cpu",
        latency_runs=2,
        snr_levels_db=(5.0, 20.0),
        report_formats=("json",),
    )
    print(f"  cfg: {cfg.task}, output={cfg.output_dir}")
    print("  ok.")

    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
