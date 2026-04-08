# Model Card: BBO-GP-Hybrid Optimisation Approach

This model card documents the **optimisation procedure** used in the BBO capstone (not a single trained neural “model” in the narrow sense). It follows the spirit of Mini-lesson 21.2: overview, intended use, technical details, performance, assumptions/limitations, and ethical / transparency considerations.

## 1. Overview

| Field | Description |
|--------|-------------|
| **Name** | BBO-GP-Hybrid |
| **Version** | 1.0 (reflects iterative refinement through **10 weekly submission rounds**; codebase and `weekly_workflow.ipynb` continue to evolve through week 12) |
| **Type** | Sequential **Bayesian optimisation-style** pipeline: surrogate regression + acquisition maximisation + optional manual overrides |
| **Core components** | `GPSurrogate` (RBF, optional ARD), `SVMSurrogate` (bootstrap ensemble for uncertainty), optional `MLPSurrogate` (MC dropout); acquisition via **UCB**, **EI**, **PI**; optimiser **random search + L-BFGS-B** with optional **regional focus** (`optimize_acquisition_with_regional_focus` in `src/acquisition.py`) |
| **Repository entry points** | `public/notebooks/weekly_workflow.ipynb`, `public/src/surrogates.py`, `public/src/acquisition.py` |

## 2. Intended use

**Suitable for**

- **Expensive black-box** objectives where each evaluation is costly and the input space is **[0, 1]^d** with **d roughly 2–8**.
- **Single-objective** maximisation (after any course-specific sign flips for “minimise harm” style objectives).
- **Low evaluation budget** (e.g. tens of points total), where hand-engineered **trust regions** and **function-specific** surrogate choice are practical.

**Should be avoided (without substantial redesign) for**

- **High dimensionality** (e.g. d ≫ 10) or **categorical / mixed** inputs — the current pipeline assumes continuous boxes and small n.
- **Stochastic** oracles with unmodelled noise — GPs use a fixed noise term; SVR uncertainty is a bootstrap heuristic, not a calibrated likelihood.
- **Multi-objective** Pareto search, **constrained** optimisation, or **safe** BO — not implemented in this baseline.
- **Real-world deployment** where errors carry legal, financial, or safety risk — this workflow is a **student capstone** on synthetic functions.

## 3. Details: strategy across ten rounds

The approach **evolved** week by week; later rounds are not the same algorithm as week 1.

| Phase | Weeks | Techniques | How the approach changed |
|--------|-------|------------|---------------------------|
| Broad exploration | W1 | UCB with moderate β per function | Establish baselines; F5 and F4 responded strongly. |
| Over-exploitation | W2 | Aggressive EI (small ξ) | **Lesson:** F5 collapsed; taught caution with ξ and boundary structure. |
| Recovery | W3 | Regional focus near W1 successes | Re-centred search; reduced catastrophic jumps. |
| Targeted exploitation | W4 | Per-function focus regions, moderate EI | Pushed known good basins (e.g. F5, F2 ridge). |
| Multi-surrogate experiment | W5 | SVR / MLP on some functions, GP-ARD elsewhere | **Lesson:** not all swaps helped; F5 regional bug underscored implementation risk. |
| Consolidation | W6 | Manual probes where flat/boundary; GP-ARD + tight focus; GP+noise for noisy 2D; SVR for 8D | Fixed focus bounds; **three new bests** on key functions. |
| Scaling / diagnostics | W7–W8 | Tighter radii; UCB vs EI trade-offs; optional **LLM-assisted** experiments (`llm_advisor.py`, logged JSON) | Emphasis on trajectory analysis and transparency tooling. |
| Conservative exploitation | W9–W10 | **Smaller focus radii**, EI for fragile peaks, **manual** boundary / grid probes where appropriate | **Lesson:** large radius + aggressive “gradient narrative” on one function **regressed**; **tiny steps** around proven bests worked better on narrow peaks. |

**Decision logic (high level):** Fit surrogate on all history → compute acquisition over candidates (global or **focused box** around anchor) → local refine → optionally replace with **manual** vector from strategy notes. **Surrogate choice** is guided by dimensionality and empirical behaviour (GP for smooth low-D; SVR ensemble for very high-D or noisy patterns).

## 4. Performance

**Metric:** **Best objective value observed so far** per function (max over all submitted points through **week 9** portal logs), plus qualitative status. Values below are taken from cumulative maxima over `public/data/results/week_1` … `week_9` outputs (and align with the updated README table).

| Function | Dims | Initial best (from course design) | Cumulative best after W9 | Status (qualitative) |
|----------|------|-----------------------------------|---------------------------|----------------------|
| F1 | 2 | ≈ 0 | ≈ 0 | Stuck / no signal |
| F2 | 2 | 0.611 | **0.621** | Stagnant after mid rounds |
| F3 | 3 | -0.035 | **-0.00553** | Near plateau |
| F4 | 4 | -4.026 | **0.667** | Narrow peak; late improvement |
| F5 | 4 | 1089 | **8662** | Solved at boundary |
| F6 | 5 | -0.714 | **-0.474** | Improving then sensitive to step size |
| F7 | 6 | 1.365 | **2.836** | Strong gains; sensitive to overshoot |
| F8 | 8 | 9.598 | **9.971** | Diminishing returns |

**Note:** Percentage improvements are meaningful only per-function (scales differ wildly). The **competition-relevant** metric is typically **rank or normalised score** on the platform, not raw y comparison across functions.

## 5. Assumptions and limitations

**Assumptions**

- **Local smoothness** in regions of interest — RBF / Matérn-style behaviour in GP; poor fit if the true landscape is sharply discontinuous.
- **Stationarity** (weak): kernel hyperparameters are shared or ARD-smoothed; **non-stationary** true functions (e.g. F7-style complexity) can mislead the posterior far from data.
- **Low effective dimension** or **factorised** relevance — ARD helps but does not guarantee identification with n < 50 in 8D.
- **Deterministic oracle** — re-querying the exact same **x** is wasted; **near** duplicates are used to probe **ridge width**, not noise.

**Limitations and failure modes**

- **Small n, high d:** surrogates become **overconfident**; acquisition can suggest misleading exterior points unless **regional focus** constrains them.
- **GP cost:** training scales **O(n³)**; beyond ~100 points, sparse approximations would be needed (not in this codebase).
- **Heuristic uncertainty (SVR bootstrap):** not calibrated; EI treats it like GP variance — **risk of over-exploration or under-exploration**.
- **Policy mistakes** (wrong radius, wrong narrative about “accelerating trends”) can dominate model error — the **human strategy layer** is part of the system.

## 6. Ethical considerations and transparency

- **Transparency:** Weekly strategies are written in `notes/BBO_Strategy_W*_Report.md` (where present); notebooks record **hyperparameters** (ξ, β, focus centre, radius, surrogate type). Another researcher can **approximate** reproduction with the same data, pinned `requirements.txt`, and fixed **NumPy random seed** (e.g. 42) where used; exact tie-breaking in L-BFGS-B and sklearn minor versions can still cause tiny numerical drift.
- **Reproducibility:** Cumulative `.npy` checkpoints and `results/week_*/inputs.txt|outputs.txt` provide an **append-only audit trail** of what was submitted and returned.
- **Real-world adaptation:** This card makes explicit that the method is **not validated** for fairness, safety, or distributional shift. Any transfer to real domains would require **problem-specific** risk analysis, constraints, and monitoring — none of which are claimed here.

**Would more detail help?** Adding acquisition heatmaps, per-round decision JSON (as in peer workflows), and pinned package versions would improve **external auditability** further; the current structure is **sufficient** for coursework handoff if the README, datasheet, strategy notes, and notebooks are kept together.
