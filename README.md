# Bayesian Black-Box Optimization Capstone Project

## Project Overview

This capstone project tackles the optimization of 8 synthetic black-box functions (2D to 8D) simulating real-world scenarios including contamination detection, drug discovery, hyperparameter tuning, and manufacturing optimization. The challenge uses Bayesian optimization under strict constraints: only 1 query per function per week, making each evaluation decision critical.

## Project Structure

```
├── src/                        # Reusable Python modules
│   ├── __init__.py             # Package exports
│   ├── data.py                 # FunctionData, load_results, initialize_from_history
│   ├── surrogates.py           # SurrogateModel ABC, GPSurrogate, SVMSurrogate, MLPSurrogate
│   ├── acquisition.py          # Acquisition functions (UCB, EI, PI) and optimizers
│   └── utils.py                # Formatting, visualization, analysis, tracking
├── notebooks/
│   ├── weekly_workflow.ipynb    # Main notebook: load data → set strategy → generate queries
│   ├── data_management.ipynb   # Load new weekly results and update function history
│   ├── model_comparison.ipynb  # Compare GP / SVR / MLP surrogates with LOO cross-validation
│   └── archive/                # Original monolithic notebooks (preserved for reference)
├── data/
│   ├── function_1/ .. function_8/   # Initial .npy inputs and outputs per function
│   └── results/
│       └── week_1/ .. week_6/       # Weekly submitted inputs.txt and received outputs.txt
├── notes/
│   ├── Function_Analysis_and_Strategy_Report.md   # Per-function analysis and Week 5 strategies
│   ├── Reflection_Responses.md                     # Written reflections on strategy evolution
│   ├── Capstone Instructions.md                    # Function descriptions and rules
│   └── Capstone project FAQ.md
└── requirements.txt            # numpy, scipy, scikit-learn, torch (CPU), matplotlib, jupyter
```

## Functions

| Function | Dims | Domain | Description |
|----------|------|--------|-------------|
| F1 | 2D | Contamination detection | Extremely localised signal in a radiation field |
| F2 | 2D | Noisy ML model | Log-likelihood with noise and multiple local optima |
| F3 | 3D | Drug discovery | Minimise adverse reactions (negated for maximisation) |
| F4 | 4D | Warehouse placement | Dynamic product placement; narrow peak with sharp cliffs |
| F5 | 4D | Chemical yield | Unimodal yield function; single peak near upper boundary |
| F6 | 5D | Recipe optimisation | Multi-criteria cake recipe scoring (flavour, cost, waste) |
| F7 | 6D | ML hyperparameters | 6 hyperparameters for an ML model; non-stationary landscape |
| F8 | 8D | High-dimensional system | 8 hyperparameters; complex interactions, hard for GP |

## Results (Weeks 1-6)

| Function | Initial Best | W1 | W2 | W3 | W4 | W5 | W6 | Cumulative Best | Improvement |
|----------|-------------|-----|-----|-----|-----|-----|-----|-----------------|-------------|
| F1 (2D) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.68e-09 | 0.000 | Stuck |
| F2 (2D) | 0.611 | 0.279 | 0.247 | 0.510 | 0.621 | 0.084 | 0.398 | 0.621 | +1.6% |
| F3 (3D) | -0.035 | -0.022 | -0.025 | -0.111 | -0.033 | **-0.006** | -0.027 | **-0.006** | +84% |
| F4 (4D) | -4.026 | 0.548 | 0.661 | -0.144 | 0.030 | 0.603 | 0.637 | 0.661 | +4.687 |
| F5 (4D) | 1088.9 | 2517.6 | 91.3 | 5328.0 | 7223.2 | 1873.8 | **8662.4** | **8662.4** | +695% |
| F6 (5D) | -0.714 | -0.565 | -0.570 | -1.501 | -0.937 | -0.811 | **-0.520** | **-0.520** | +27% |
| F7 (6D) | 1.365 | 1.711 | 1.597 | 2.468 | 1.585 | 0.947 | 2.284 | 2.468 | +81% |
| F8 (8D) | 9.598 | 9.515 | 9.932 | 9.828 | 9.224 | 9.887 | **9.947** | **9.947** | +3.6% |

## Technical Approach

### Surrogate Models (`src/surrogates.py`)

All surrogates implement a common `SurrogateModel` ABC with `fit(X, y)`, `predict(X) -> (mean, std)`, and `get_name()`, making them interchangeable in the acquisition pipeline.

- **GPSurrogate** — Gaussian Process with RBF kernel. Supports ARD (Automatic Relevance Determination) for per-dimension length scales. Native uncertainty via posterior variance. Best for low-dimensional functions with smooth landscapes (F3, F4, F5).
- **SVMSurrogate** — Support Vector Regression (RBF kernel) with bootstrap ensemble (default 20 models) for uncertainty estimation. Epsilon-insensitive loss naturally filters noise. Recommended for noisy functions (F2) and high-dimensional problems (F8).
- **MLPSurrogate** — PyTorch neural network (default 64-32 hidden layers, ReLU, dropout) with MC Dropout for uncertainty at prediction time (50 forward passes). Captures non-stationary surfaces and complex interactions. Recommended for multi-criteria (F6), hyperparameter (F7), and high-dimensional (F8) functions.

### Acquisition Functions (`src/acquisition.py`)

- **UCB** (Upper Confidence Bound): `mean + β * std` — adjustable exploration via β parameter
- **EI** (Expected Improvement): probabilistic improvement over current best — controlled by ξ
- **PI** (Probability of Improvement): probability of exceeding current best

Optimization uses a two-stage hybrid: random candidate generation (with optional regional focus near a target point) followed by L-BFGS-B local refinement. When regional focus is used, L-BFGS-B bounds are constrained to the focus box so the optimizer cannot escape the target region.

### Strategy Evolution

| Phase | Weeks | Approach | Key Lesson |
|-------|-------|----------|------------|
| Exploration | W1 | UCB with moderate β (1.5-3.0) per function | F5 +131%, F4 turned positive; F1 stuck at zero |
| Over-exploitation | W2 | Aggressive EI (ξ=0.001-0.005) | F5 collapsed 96%; most functions declined |
| Recovery | W3 | Regional focus near W1 successes, increased exploration | F5 recovered to 5328 (+112%); F7 new best |
| Targeted | W4 | Function-specific focus regions, moderate exploitation | F5 hit 7223; F2 recovered to initial best |
| Multi-model | W5 | SVR for F2, MLP for F6-F8, GP-ARD for F3-F4 | Only F3 improved; SVR/MLP regressions; F5 escaped focus (bug) |
| Consolidation | W6 | Manual F1/F5; GP-ARD for F3-F4,F6-F7; GP+noise F2; SVR F8; regional focus bounds fixed | 3 new bests (F5, F6, F8); F5 +20%, F6 +8%, F8 +0.2% |
| Final push | W7 | Manual F1/F5; GP tight for F2 (new point near best); GP-ARD tighter (F3/F4/F6/F7); focus on W6 bests for F6/F8 | Exploit W6 wins; F2: new query near [0.70, 0.93]; probe F1 near center |

### Key Insights

1. **Dimensionality determines surrogate choice**: GP works well up to ~4D; MLP or SVR needed for 5D+
2. **Data scarcity demands caution**: With 14-44 points, aggressive exploitation risks overfitting to sparse data
3. **Function-specific strategies are essential**: F5 (unimodal, push boundary) vs F4 (narrow peak, ultra-tight focus) vs F1 (no signal, space-filling) require fundamentally different approaches
4. **Regional focus prevents catastrophic jumps**: Focus radius constraints (0.02-0.15) keep queries near known good regions, avoiding cliffs like F4's Week 3 collapse
5. **Failure teaches more than success**: Week 2's 96% F5 collapse led to the recovery strategy and regional focus mechanism used in all subsequent weeks

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For PyTorch CPU-only (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Weekly Workflow

1. **Load results**: Run `notebooks/data_management.ipynb` to import new weekly outputs
2. **Compare models**: Optionally run `notebooks/model_comparison.ipynb` to evaluate surrogates via LOO cross-validation
3. **Generate queries**: Run `notebooks/weekly_workflow.ipynb` — edit the strategy dict, generate queries, format for portal submission

---

**Detailed per-function analysis**: See [`notes/Function_Analysis_and_Strategy_Report.md`](notes/Function_Analysis_and_Strategy_Report.md)
**Technical justification and literature**: See [`notes/Technical_Justification.md`](notes/Technical_Justification.md)
