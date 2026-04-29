"""Build the cumulative-best progress plot for the README.

Parses ``public/data/results/week_*/outputs.txt`` and ``inputs.txt`` to extract
per-week observations, computes the cumulative best per function, and saves a
2x4 grid of subplots (one per function) plus a normalised overlay plot.

Run once per refresh:
    python public/figures/_build_cumulative_best.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "public" / "data" / "results"
FIGURES_DIR = REPO_ROOT / "public" / "figures"

# Pattern captures every `np.float64(<value>)` entry in the portal log files.
FLOAT_PATTERN = re.compile(r"np\.float64\(([^)]+)\)")

FUNCTION_LABELS = [
    "F1 (2D · contamination)",
    "F2 (2D · noisy ML)",
    "F3 (3D · drug discovery)",
    "F4 (4D · warehouse)",
    "F5 (4D · chemical yield)",
    "F6 (5D · recipe)",
    "F7 (6D · ML hypers)",
    "F8 (8D · high-d system)",
]

# Initial-design bests supplied by the course (from prior README).
INITIAL_BEST = np.array([0.0, 0.611, -0.035, -4.026, 1089.0, -0.714, 1.365, 9.598])


def _load_weekly_outputs(results_dir: Path) -> np.ndarray:
    """Return an (n_weeks, 8) array of per-week observed values, ordered by week.

    The latest ``week_N/outputs.txt`` file is cumulative (one row per week, eight
    floats per row), so parsing that single file is sufficient and avoids the
    duplicate-row problem in intermediate files.
    """
    week_dirs = sorted(
        (d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("week_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not week_dirs:
        raise FileNotFoundError(f"No week_* directories under {results_dir}")

    latest = week_dirs[-1] / "outputs.txt"
    flat = [float(m.group(1)) for m in FLOAT_PATTERN.finditer(latest.read_text())]
    if len(flat) % 8 != 0:
        raise ValueError(f"{latest}: expected a multiple of 8 values, got {len(flat)}")
    return np.asarray(flat, dtype=float).reshape(-1, 8)


def _cumulative_best(per_week: np.ndarray) -> np.ndarray:
    """Cumulative best per function, assuming maximisation for all (after sign flips)."""
    running = np.copy(per_week)
    for i in range(1, running.shape[0]):
        running[i] = np.maximum(running[i], running[i - 1])
    return running


def main() -> None:
    per_week = _load_weekly_outputs(RESULTS_DIR)
    n_weeks = per_week.shape[0]
    weeks = np.arange(1, n_weeks + 1)

    cum_best = _cumulative_best(per_week)

    # Prepend the initial best (week 0) so the baseline is visible.
    weeks_with_init = np.concatenate([[0], weeks])
    cum_best_with_init = np.vstack([INITIAL_BEST, cum_best])

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"Cumulative best per function across weeks W1–W{n_weeks} (initial design at W0)",
        fontsize=14,
        y=0.995,
    )

    for i, ax in enumerate(axes.flat):
        ax.plot(
            weeks_with_init,
            cum_best_with_init[:, i],
            marker="o",
            linewidth=2,
            color="tab:blue",
        )
        ax.scatter(
            weeks,
            per_week[:, i],
            marker="x",
            color="tab:orange",
            s=35,
            label="weekly query",
            zorder=3,
        )
        ax.set_title(FUNCTION_LABELS[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Week")
        if i % 4 == 0:
            ax.set_ylabel("Objective value (max)")

        # Annotate the current best.
        best_val = cum_best_with_init[-1, i]
        best_week_idx = int(np.argmax(cum_best_with_init[:, i]))
        ax.annotate(
            f"best={best_val:.3g} @ W{best_week_idx}",
            xy=(best_week_idx, best_val),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color="tab:blue",
        )

    handles = [
        plt.Line2D([0], [0], color="tab:blue", marker="o", label="cumulative best"),
        plt.Line2D(
            [0], [0], color="tab:orange", marker="x", linestyle="", label="weekly query"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_path = FIGURES_DIR / "cumulative_best.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
