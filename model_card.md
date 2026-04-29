# Model Card: BBO-GP-Hybrid Optimisation Approach

This card documents the **optimisation procedure** used in the Imperial College BBO capstone. It is not a single trained neural "model" in the narrow sense but a sequential Bayesian-optimisation pipeline whose decisions are driven by per-function surrogates, acquisition functions, and a human strategy layer. Structure follows the course template: Model Description (Input / Output / Model Architecture), Performance, Limitations, Trade-offs.

## Overview

| Field | Description |
|-------|-------------|
| **Name** | BBO-GP-Hybrid |
| **Version** | 1.1 (13 weekly submission rounds completed; W11 added cluster-informed anchors, W12 added PCA-derived focus regions, W13 applied PCA plan and produced a new best on F4) |
| **Type** | Sequential **Bayesian-optimisation-style** pipeline: surrogate regression + acquisition maximisation + optional manual overrides |
| **Core components** | `GPSurrogate` (RBF, optional ARD), `SVMSurrogate` (bootstrap ensemble for uncertainty), optional `MLPSurrogate` (MC Dropout); acquisition via **UCB**, **EI**, **PI**; optimiser **random search + L-BFGS-B** with optional **regional focus** (`optimize_acquisition_with_regional_focus` in [`src/acquisition.py`](src/acquisition.py)) |
| **Repository entry points** | [`notebooks/weekly_workflow.ipynb`](notebooks/weekly_workflow.ipynb), [`notebooks/pca_analysis.ipynb`](notebooks/pca_analysis.ipynb), [`src/surrogates.py`](src/surrogates.py), [`src/acquisition.py`](src/acquisition.py) |

## Model Description

### Input

Per call at week *k* for function `fᵢ`, the pipeline consumes:

- The cumulative observation history `X ∈ ℝ^{n × dᵢ}`, `y ∈ ℝ^n` where `n = n_initial_i + k` grows by one each week.
- A **strategy specification** (dict in `weekly_workflow.ipynb`): surrogate type, acquisition type and hyperparameter (`β` for UCB, `ξ` for EI), optional focus centre `x₀ ∈ [0,1]^dᵢ`, optional focus radius `r > 0`, and optional manual-override vector.
- Global bounds `[0, 1]^dᵢ` (unit hypercube) for every function.

### Output

A **single query vector** `x* ∈ [0, 1]^dᵢ` per function per week, formatted for portal submission by `format_for_portal()` (`src/utils.py`). The portal returns one scalar `yᵢ = fᵢ(x*)` per function, appended to the history for the next round. The cumulative by-function `best_y` after *k* rounds is the primary reported metric.

### Model Architecture

Three interchangeable surrogates live behind a common `SurrogateModel` ABC with `fit(X, y)`, `predict(X) -> (mean, std)`, and `get_name()`:

| Surrogate | Implementation | Uncertainty source | Typical functions |
|-----------|----------------|--------------------|-------------------|
| **GPSurrogate** | `sklearn.gaussian_process.GaussianProcessRegressor` with RBF (or anisotropic-RBF / ARD) kernel, `alpha` for noise, 10 restarts of the marginal-likelihood optimiser | Posterior variance | F2, F3, F4, F5 |
| **SVMSurrogate** | `sklearn.svm.SVR(kernel='rbf')` inside a 20-model bootstrap ensemble over the training set | Ensemble std-deviation of predictions | F2 early, F8 |
| **MLPSurrogate** | PyTorch `nn.Sequential(Linear→ReLU→Dropout)` × 2 hidden (64, 32), Adam `lr=1e-3`, dropout `p=0.1`, 50 MC-Dropout passes at inference | MC-Dropout predictive std | Considered for F6, F7 |

Each surrogate feeds one of three **acquisition functions** in [`src/acquisition.py`](src/acquisition.py):

- **UCB**: `μ(x) + β · σ(x)`, where `β ∈ [1.5, 3.0]` tunes exploration;
- **EI**: expected improvement over current best with margin `ξ ∈ [1e-4, 5e-3]`;
- **PI**: probability of improvement (used sparingly; tends to over-exploit).

Acquisition maximisation is a **two-stage hybrid**: 10 000 – 20 000 random candidates inside the (optionally focus-constrained) unit box, followed by **L-BFGS-B** refinement of the top candidates. `optimize_acquisition_with_regional_focus` adds an axis-aligned box constraint `|x − x₀|_∞ ≤ r` around a hand-picked anchor `x₀`; L-BFGS-B bounds are clipped to this box so the optimiser cannot escape the focus region.

From **W12 onward**, focus anchors are derived from [`notebooks/pca_analysis.ipynb`](notebooks/pca_analysis.ipynb): linear PCA (and RBF-kernel PCA where non-linear structure is plausible) of the top-K best points per function defines the exploitation axis, and low-variance dimensions are effectively frozen by shrinking `r`. Outputs are cached in [`data/results/week_12/pca_week12.json`](data/results/week_12/pca_week12.json).

Final decisions in any week may be overridden by a **manual vector** from the corresponding strategy report in [`../notes/`](../notes/) — this human layer is an explicit part of the architecture, not a bypass.

## Intended Use

**Suitable for**
- **Expensive black-box** objectives in **[0, 1]^d** with `d` between roughly 2 and 8.
- **Single-objective** maximisation after any course-specific sign flips (F3 minimises adverse reactions; its outputs are negated).
- **Very low evaluation budget** (tens of points), where per-function surrogate choice and hand-designed trust regions are practical.

**Should be avoided (without substantial redesign) for**
- **High dimensionality** (`d ≫ 10`) or **categorical / mixed** inputs — the pipeline assumes continuous boxes and small `n`.
- **Stochastic oracles** with unmodelled noise — GPs use a fixed noise term and the SVR uncertainty is a bootstrap heuristic, not a calibrated likelihood.
- **Multi-objective** Pareto search, **constrained** optimisation, or **safe** BO — none is implemented.
- **Real-world deployment** carrying legal, financial, or safety risk — this is a coursework baseline on synthetic functions.

## Strategy Evolution — All Thirteen Rounds

The approach is not stationary: later rounds add mechanisms that were not present in week 1.

| Phase | Weeks | Techniques | What changed |
|-------|-------|------------|--------------|
| Broad exploration | W1 | UCB with moderate β per function | Baselines set; F5 (+131 %) and F4 responded strongly |
| Over-exploitation | W2 | Aggressive EI (ξ = 1e-3 – 5e-3) | F5 collapsed 96 %; taught caution on boundary structure |
| Recovery | W3 | Regional focus near W1 successes | Re-centred search; F5 recovered to 5328; F7 new best |
| Targeted exploitation | W4 | Per-function focus regions, moderate EI | F5 → 7223; F2 recovered to initial best (0.621) |
| Multi-surrogate experiment | W5 | SVR for F2; MLP for F6–F8; GP-ARD elsewhere | Only F3 improved; MLP/SVR regressions; F5 focus-bounds bug |
| Consolidation | W6 | Manual F1/F5; GP-ARD + tight focus; GP + noise for noisy 2D; SVR for 8D; focus-bounds bug fixed | Three new bests (F5 = 8662, F6, F8); **F1 central probe `[0.5, 0.5]` produced the largest non-trivial F1 value observed** |
| Final push | W7 | Manual F1/F5; GP tight for F2; GP-ARD tighter on F3–F7; W6 bests as anchors for F6/F8 | F7 / F8 new bests |
| LLM + diagnostics | W8 | Surrogate defaults plus optional LLM-assisted experiments; deliberate low-corner probe on one function | Structured JSON logging; global-structure vs boundary-peak test |
| Scaling / mixed | W9 | Manual re-centring on narrow peaks and boundary; UCB on one trajectory; SVR micro-step on 8D | Large radius can regress; tiny steps help on sharp peaks |
| Conservative | W10 | Smaller radii; EI over UCB where W9 regressed; surrogate micro-perturb near W9 bests | F7 / F8 new bests; cumulative bests protected |
| Clustering | W11 | K-means cluster analysis; trajectory extrapolation (F7); baker's-ratio domain prior (F6); random probe (F1) | F7 new best (3.114); 6 regressions; deterministic-oracle trap on F5 |
| PCA-informed | W12 | PCA / kernel-PCA anchors per function; PC1 extrapolation for F7 (α = 1.0, β = 1.8); 1-D manifold confirmed for F8 | **F2 new best (0.706); F7 new best (3.233)** |
| Final shot | W13 | Applied W12 PCA plan; ultra-tight r = 0.005 for F4; central re-probe on F1 at `[⅓, ⅓]` | **F4 new best (0.673)**; F1 central-probe signal replicated; F5/F7/F8 effectively tied at ceiling |

**Decision logic (high level):** fit surrogate on all history → compute acquisition over candidates (global or focus-box) → local refine → optionally replace with manual vector from strategy notes. Surrogate choice is guided by dimensionality and empirical behaviour.

## Performance

**Metric.** Best objective value observed so far per function (max over all submitted points through **week 13**), plus qualitative status. Values are taken from [`data/results/week_13/outputs.txt`](data/results/week_13/outputs.txt) and match the README scoreboard.

| Function | Dims | Initial best | Cumulative best after W13 | Best week | Status |
|----------|:----:|-------------:|-------------------------:|:---------:|--------|
| F1 | 2 | ≈ 0 | `2.68e−09` | W6 | Central-probe signal; W13 replicated at `[⅓, ⅓]` |
| F2 | 2 | 0.611 | **0.706** | **W12** | New best in W12; W13 regressed 29.9 % |
| F3 | 3 | −0.035 | **−0.00553** | W5 | Near plateau; W13 at −0.016 |
| F4 | 4 | −4.026 | **0.673** | **W13** | **New best in W13** |
| F5 | 4 | 1089 | **8662** | W6 / W9 / W13 | Solved at `[1, 1, 1, 1]` corner |
| F6 | 5 | −0.714 | **−0.474** | W8 | Four consecutive regressions W10–W13 |
| F7 | 6 | 1.365 | **3.233** | **W12** | Accelerating W7–W12; plateaued W13 (3.227) |
| F8 | 8 | 9.598 | **9.972** | W10 / W11 / W12 | Saturated; diminishing returns |

Cumulative-best progression is visualised in [`figures/cumulative_best.png`](figures/cumulative_best.png). Cross-function comparison of raw `y` is not meaningful without normalisation; the competition-relevant metric on the course platform is **rank or normalised score** across functions, not raw value.

## Limitations

**Assumptions**
- **Local smoothness** in regions of interest — RBF / Matérn behaviour in the GP; poor fit under sharp discontinuities.
- **Weak stationarity** — kernel hyperparameters shared across the domain or ARD-smoothed; non-stationary truth (F7-style) can mislead the posterior far from data.
- **Low effective dimension** or **factorised relevance** — ARD helps but does not guarantee identification with `n < 50` in 8D.
- **Deterministic oracle** — re-querying the exact same `x` is wasted; near-duplicates probe ridge width, not noise.

**Failure modes observed across the 13 rounds**
- **Surrogate over-confidence at small `n`, high `d`**: acquisition can suggest misleading exterior points unless regional focus constrains them (W2 F5 collapse, W3 F4 collapse).
- **GP cubic cost** `O(n³)`: fine for `n ≤ 100` but would need sparse approximations for longer horizons.
- **Uncalibrated SVR uncertainty**: EI treats the bootstrap std like a GP variance — risks over- or under-exploration.
- **Narrative bias in the human layer**: the "accelerating trend" story on F7 in W9 used a radius that was too large and regressed; W10 corrected with a tighter radius and EI.
- **Prematurely declaring a function "stuck"**: F1 was labelled flat for ten rounds; W6 and W13 central probes both produced signals many orders of magnitude above every boundary query, retrospectively invalidating the "no signal" narrative.

## Trade-offs

The pipeline makes several explicit compromises; each has a context where it hurts.

| Trade-off | Design choice | When it helps | When it hurts |
|-----------|---------------|---------------|---------------|
| **Exploration vs exploitation** | UCB β and EI ξ set per function, per week | Tight β / ξ on near-ceiling functions protects cumulative best (F3, F4, F8) | Aggressive ξ early (W2) collapsed F5 by 96 %; wide β on narrow peaks overshoots (F4 W3: −0.144) |
| **Surrogate fidelity vs training cost** | GP up to 4D, SVR / MLP in higher D | GP gives calibrated variance where it fits (F2–F5) | GP posterior is miscalibrated on non-stationary F7; MLP needed more data than available in W5 |
| **Regional focus vs global search** | Axis-aligned focus box with radius `r` around anchor | Prevents catastrophic escapes (every round after W3) | Can lock the search inside a suboptimal basin if the anchor is wrong (F6 W10–W13 regressions sit inside the wrong basin) |
| **Automated acquisition vs human override** | Manual vector can replace acquisition recommendation | Incorporates domain knowledge (F6 baker's ratios, F5 corner probe, F1 max-min distance) | Introduces narrative bias (F7 W9) and subjectivity that the audit trail must capture |
| **Sign-flipped maximisation on F3** | All objectives converted to maximisation | Unifies the acquisition code path | Obscures the domain meaning (smaller absolute values are "better") for non-technical readers |
| **Deterministic oracle assumption** | Kernel noise set small; no re-queries | Avoids wasting weeks on duplicates | Masks genuine oracle fluctuations if they exist (e.g. F5 W6 vs W9 at the corner: 8662.405 vs 8662.405 — consistent) |
| **PCA anchors from best-K clusters** | PC1 of top-K points defines exploitation axis | Identified an effective-1D manifold on F8 and the migrating trajectory on F7 | A single outlier (F6 W3 at −1.50) can dominate PC1 and invert sign conventions — we mitigate with Pearson `r` cross-checks |

## Ethical Considerations and Transparency

- **Transparency.** Weekly strategies are written in `../notes/BBO_Strategy_W*_Report.md` (W7, W8, W9, W11, W12 present). Notebooks record hyperparameters (`β`, `ξ`, focus centre, radius, surrogate type). Another researcher can approximate reproduction with the same data, pinned `requirements.txt`, and a fixed NumPy random seed (e.g. 42). Exact tie-breaking in L-BFGS-B and sklearn minor-version drift can still cause tiny numerical differences.
- **Reproducibility.** Cumulative `.npy` checkpoints plus [`data/results/week_*/inputs.txt` and `outputs.txt`](data/results/) provide an append-only audit trail of what was submitted and returned.
- **Real-world adaptation.** This card makes explicit that the method is **not validated** for fairness, safety, or distributional shift. Any transfer to real domains would require problem-specific risk analysis, constraints, and monitoring — none of which are claimed here.

**Improvements that would raise auditability.** Per-round decision JSON (already present for the W8 LLM experiments and W12 PCA, not yet for other weeks), acquisition heatmaps, and pinned package versions with a `pip freeze` of the exact 13-week environment.
