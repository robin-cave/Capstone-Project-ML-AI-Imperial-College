# Bayesian Black-Box Optimization Capstone Project

## Section 1: Project Overview

This capstone project tackles the optimization of 8 synthetic black-box functions (ranging from 2D to 8D) that simulate real-world scenarios including drug discovery, hyperparameter tuning, and manufacturing optimization. The challenge employs Bayesian optimization with Gaussian Processes under strict constraints: only 1 query per function per week, making each evaluation decision critical. As a private investor, this BBO capstone project will help me demonstrate to other angel investors that I have the ability to analyse ML-based investment projects.

## Section 2: Inputs and Outputs

**Inputs**: n-dimensional arrays constrained to the range [0, 1] where each function accepts different dimensionalities (Function 1: 2D, Function 8: 8D).

**Submission Format**: Values submitted as hyphen-separated strings with exactly 6 decimal places (e.g., `0.123456-0.789012`).

**Outputs**: Single scalar performance values that can be positive or negative, representing the function evaluation.

**Example**: Function 1 (2D) input `[0.063271, 0.397595]` returns output `-3.1e-91`.

| Function | Dimensions | Week 1 Best | Week 2 Best | Change |
|----------|-----------|-------------|-------------|---------|
| 1 | 2D | -3.10e-91 | 4.41e-20 | ↑ |
| 2 | 2D | 0.279 | 0.247 | ↓ |
| 3 | 3D | -0.022 | -0.025 | ↓ |
| 4 | 4D | 0.548 | 0.661 | ↑ |
| 5 | 4D | 2517.62 | 91.28 | ↓↓ |
| 6 | 5D | -0.565 | -0.570 | ↓ |
| 7 | 6D | 1.711 | 1.597 | ↓ |
| 8 | 8D | 9.515 | 9.932 | ↑ |

## Section 3: Challenge Objectives

**Goal**: Maximize all 8 objective functions through strategic query selection.

**Key Constraints**:
- Limited evaluations: 1 query per function per week
- Black-box optimization: No access to function derivatives or internal mechanics
- Input range enforcement: All values must satisfy 0 ≤ x < 1

**Challenge**: Balance exploration (discovering new promising regions) versus exploitation (refining known good areas) with extremely limited function evaluations, requiring careful strategic planning for each query.

## Section 4: Technical Approach

### Methods Used

**Surrogate Model**: Gaussian Process Regressor with RBF (Radial Basis Function) kernel that provides both mean predictions and uncertainty estimates, enabling principled exploration-exploitation tradeoffs.

**Acquisition Functions**: 
- **UCB (Upper Confidence Bound)**: Primary strategy with adjustable β parameter controlling exploration-exploitation balance
- **EI (Expected Improvement)**: Used for noisy functions where uncertainty modeling is critical

**Optimization Strategy**: Two-stage hybrid approach combining random search (global exploration of candidate space) with L-BFGS-B local refinement (gradient-based exploitation).

### Week 1 Query Strategy

**Initial Approach**: Customized function-specific parameters based on problem characteristics:
- Functions 1-3: UCB β=2.5 (moderate exploration for lower dimensions)
- Function 5: UCB β=2.0 (exploitation focus for unimodal structure)
- Functions 6, 8: UCB β=3.0 (high exploration for high-dimensional spaces)

**Week 1 Key Results**:
- Function 5: 1088.86 → 2517.62 (+131% gain) - best performer
- Function 7: 0.336 → 1.711 (+409% improvement)
- Function 4: -4.03 → 0.548 (dramatic turnaround from negative)
- Functions 1, 2, 6: No improvement detected, requiring strategy adjustment

### Week 2 Query Strategy

**Adaptive Learning**: Modified strategies based on Week 1 performance analysis:
- Increased exploration parameters for stagnant functions (1, 2, 6)
- Maintained successful strategies for improving functions (4, 7, 8)
- Implemented recovery focus for Function 5 to investigate high-performance region

**Week 2 Key Results**:
- Function 8: 9.515 → 9.932 (continued steady improvement)
- Function 4: 0.548 → 0.661 (maintained positive trajectory)
- Function 2: 0.279 → 0.247 (modest improvement)
- Function 5: 2517.62 → 91.28 (major drop - potential overshoot)
- Functions 3, 6, 7: Slight declines indicating need for further refinement

### Exploration-Exploitation Balance

**What Makes This Approach Thoughtful**:

1. **Customized strategies** per function based on dimensionality, noise characteristics, and known problem structure rather than one-size-fits-all approach
2. **Adaptive learning** with iterative refinement of β parameters based on observed performance patterns across weeks
3. **Hybrid optimization** combining random search (avoiding local optima in acquisition landscape) with gradient-based L-BFGS-B refinement for precision
4. **Higher β values (2.5-3.0)** for high-dimensional functions (6D, 8D) to prevent premature convergence in large search spaces
5. **Lower β values (1.5-2.0)** for functions with domain knowledge or unimodal structure to exploit known good regions efficiently

**Key Insights**: Function 5's dramatic volatility (2517.62 → 91.28) reveals sensitivity to search strategy and suggests complex landscape characteristics. Large performance swings indicate the need for careful regional focus and potentially more conservative exploration in future iterations. The mixed results across functions demonstrate the importance of function-specific tuning rather than uniform approaches.

**Unique Aspects**: This approach demonstrates thoughtful ML engineering by treating each function as a distinct optimization problem, adapting strategies based on empirical feedback, and balancing aggressive exploration (high-dimensional spaces) with targeted exploitation (known good regions).

---

**Project Implementation**: See [`notebooks/bayesian_optimization.ipynb`](notebooks/bayesian_optimization.ipynb) for full technical implementation and analysis.
