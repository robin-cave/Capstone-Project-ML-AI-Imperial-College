"""
Acquisition functions and optimization for Bayesian optimization.
"""
from typing import Any, Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .surrogates import SurrogateModel
from .data import FunctionData


class AcquisitionFunction:
    """Collection of acquisition functions for Bayesian optimization."""

    @staticmethod
    def ucb(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound: UCB = mean + beta * std"""
        return mean + beta * std

    @staticmethod
    def ei(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement"""
        std = np.maximum(std, 1e-9)
        z = (mean - y_best - xi) / std
        return (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def pi(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement"""
        std = np.maximum(std, 1e-9)
        z = (mean - y_best - xi) / std
        return norm.cdf(z)


def optimize_acquisition(
    surrogate: SurrogateModel,
    func_data: FunctionData,
    acq_func: str = "ucb",
    n_random: int = 1000,
    n_refine: int = 10,
    **acq_params: Any,
) -> np.ndarray:
    """Find the point that maximizes the acquisition function."""
    acq_func = acq_func.lower()
    n_dims = func_data.n_dims
    bounds = []
    for i in range(n_dims):
        x_min, x_max = func_data.inputs[:, i].min(), func_data.inputs[:, i].max()
        margin = (x_max - x_min) * 0.1
        bounds.append((max(0, x_min - margin), min(1, x_max + margin)))
    X_random = np.random.uniform(0, 1, size=(n_random, n_dims))
    for i in range(n_dims):
        X_random[:, i] = X_random[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    mean, std = surrogate.predict(X_random)
    _, y_best = func_data.get_best()
    if acq_func == "ucb":
        acq_values = AcquisitionFunction.ucb(mean, std, beta=acq_params.get("beta", 2.0))
    elif acq_func == "ei":
        acq_values = AcquisitionFunction.ei(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    elif acq_func == "pi":
        acq_values = AcquisitionFunction.pi(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func}")
    best_candidates_idx = np.argsort(acq_values)[-n_refine:]
    best_candidate = None
    best_acq_value = -np.inf
    for idx in best_candidates_idx:
        x0 = X_random[idx]

        def objective(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            m, s = surrogate.predict(x)
            if acq_func == "ucb":
                return -float(AcquisitionFunction.ucb(m, s, beta=acq_params.get("beta", 2.0))[0])
            if acq_func == "ei":
                return -float(AcquisitionFunction.ei(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])
            return -float(AcquisitionFunction.pi(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if -result.fun > best_acq_value:
            best_acq_value = -result.fun
            best_candidate = result.x
    return best_candidate


def optimize_acquisition_enhanced(
    surrogate: SurrogateModel,
    func_data: FunctionData,
    acq_func: str = "ucb",
    n_random: int = 1000,
    n_refine: int = 10,
    bound_margin: float = 0.02,
    expand_search: bool = True,
    **acq_params: Any,
) -> Tuple[np.ndarray, float, float]:
    """Enhanced acquisition optimization with boundary enforcement. Returns (best_point, pred_mean, pred_std)."""
    acq_func = acq_func.lower()
    n_dims = func_data.n_dims
    n_random_scaled = min(n_random * (1 + n_dims // 4), 6000)
    if expand_search:
        bounds = [(bound_margin, 1.0 - bound_margin) for _ in range(n_dims)]
    else:
        bounds = []
        for i in range(n_dims):
            x_min, x_max = func_data.inputs[:, i].min(), func_data.inputs[:, i].max()
            margin = (x_max - x_min) * 0.2
            bounds.append((max(bound_margin, x_min - margin), min(1.0 - bound_margin, x_max + margin)))
    X_random = np.zeros((n_random_scaled, n_dims))
    for i in range(n_dims):
        X_random[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], n_random_scaled)
    best_x, _ = func_data.get_best()
    n_near_best = n_random_scaled // 5
    X_near_best = np.clip(best_x + np.random.normal(0, 0.1, size=(n_near_best, n_dims)), bound_margin, 1.0 - bound_margin)
    X_random = np.vstack([X_random, X_near_best])
    mean, std = surrogate.predict(X_random)
    _, y_best = func_data.get_best()
    if acq_func == "ucb":
        acq_values = AcquisitionFunction.ucb(mean, std, beta=acq_params.get("beta", 2.0))
    elif acq_func == "ei":
        acq_values = AcquisitionFunction.ei(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    elif acq_func == "pi":
        acq_values = AcquisitionFunction.pi(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func}")
    best_candidates_idx = np.argsort(acq_values)[-n_refine:]
    best_candidate = X_random[best_candidates_idx[-1]].copy()
    best_acq_value = -np.inf
    for idx in best_candidates_idx:
        x0 = X_random[idx]

        def objective(x: np.ndarray) -> float:
            x = np.clip(x, bound_margin, 1.0 - bound_margin)
            m, s = surrogate.predict(x.reshape(1, -1))
            if acq_func == "ucb":
                return -float(AcquisitionFunction.ucb(m, s, beta=acq_params.get("beta", 2.0))[0])
            if acq_func == "ei":
                return -float(AcquisitionFunction.ei(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])
            return -float(AcquisitionFunction.pi(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if -result.fun > best_acq_value:
            best_acq_value = -result.fun
            best_candidate = np.clip(result.x, bound_margin, 1.0 - bound_margin)
    pred_mean, pred_std = surrogate.predict(best_candidate.reshape(1, -1))
    return best_candidate, float(pred_mean[0]), float(pred_std[0])


def optimize_acquisition_with_regional_focus(
    surrogate: SurrogateModel,
    func_data: FunctionData,
    acq_func: str = "ucb",
    n_random: int = 1000,
    bound_margin: float = 0.02,
    expand_search: bool = True,
    focus_region: Optional[np.ndarray] = None,
    focus_radius: float = 0.15,
    random_state: Optional[int] = None,
    **acq_params: Any,
) -> Tuple[np.ndarray, float, float]:
    """Enhanced acquisition optimization with optional regional focus. Returns (next_query, pred_mean, pred_std).
    When focus_region is set, L-BFGS-B bounds are constrained to the focus box so the optimizer cannot escape."""
    rng = np.random.default_rng(random_state)
    n_dims = func_data.n_dims
    n_refine = 10
    if expand_search:
        full_bounds = [(bound_margin, 1.0 - bound_margin) for _ in range(n_dims)]
    else:
        X_min = np.clip(func_data.inputs.min(axis=0) - 0.1, bound_margin, 1.0 - bound_margin)
        X_max = np.clip(func_data.inputs.max(axis=0) + 0.1, bound_margin, 1.0 - bound_margin)
        full_bounds = list(zip(X_min, X_max))
    bounds = full_bounds
    if focus_region is not None:
        focus_bounds = [
            (max(bound_margin, float(focus_region[i]) - focus_radius), min(1.0 - bound_margin, float(focus_region[i]) + focus_radius))
            for i in range(n_dims)
        ]
        bounds = focus_bounds
    X_random = rng.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_random, n_dims)
    )
    if focus_region is not None:
        n_focus = n_random // 3
        low_focus = np.array([b[0] for b in focus_bounds])
        high_focus = np.array([b[1] for b in focus_bounds])
        X_focus = np.clip(focus_region + rng.normal(0, focus_radius, (n_focus, n_dims)), low_focus, high_focus)
        X_random = np.vstack([X_random, X_focus])
    best_x, _ = func_data.get_best()
    n_near_best = n_random // 5
    near_best_scale = min(0.1, focus_radius) if focus_region is not None else 0.1
    X_near_best = np.clip(best_x + rng.normal(0, near_best_scale, (n_near_best, n_dims)), bound_margin, 1.0 - bound_margin)
    X_random = np.vstack([X_random, X_near_best])
    mean, std = surrogate.predict(X_random)
    _, y_best = func_data.get_best()
    if acq_func == "ucb":
        acq_values = AcquisitionFunction.ucb(mean, std, beta=acq_params.get("beta", 2.0))
    elif acq_func == "ei":
        acq_values = AcquisitionFunction.ei(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    elif acq_func == "pi":
        acq_values = AcquisitionFunction.pi(mean, std, y_best, xi=acq_params.get("xi", 0.01))
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func}")
    best_candidates_idx = np.argsort(acq_values)[-n_refine:]
    if focus_region is not None:
        low = np.array([b[0] for b in focus_bounds])
        high = np.array([b[1] for b in focus_bounds])
        in_focus = np.all((X_random >= low) & (X_random <= high), axis=1)
        focus_idx = np.where(in_focus)[0]
        best_in_focus = [i for i in best_candidates_idx if in_focus[i]]
        if len(best_in_focus) == 0:
            best_candidates_idx = focus_idx[np.argsort(acq_values[focus_idx])[-n_refine:]] if len(focus_idx) >= n_refine else focus_idx
        else:
            best_in_focus_sorted = sorted(best_in_focus, key=lambda i: acq_values[i])[-n_refine:]
            best_candidates_idx = np.array(best_in_focus_sorted)
    best_candidate = X_random[best_candidates_idx[-1]].copy()
    best_acq_value = -np.inf
    for idx in best_candidates_idx:
        x0 = X_random[idx]

        def objective(x: np.ndarray) -> float:
            x = np.clip(x, bound_margin, 1.0 - bound_margin).reshape(1, -1)
            m, s = surrogate.predict(x)
            if acq_func == "ucb":
                return -float(AcquisitionFunction.ucb(m, s, beta=acq_params.get("beta", 2.0))[0])
            if acq_func == "ei":
                return -float(AcquisitionFunction.ei(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])
            return -float(AcquisitionFunction.pi(m, s, y_best, xi=acq_params.get("xi", 0.01))[0])

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if -result.fun > best_acq_value:
            best_acq_value = -result.fun
            best_candidate = np.clip(result.x, bound_margin, 1.0 - bound_margin)
    pred_mean, pred_std = surrogate.predict(best_candidate.reshape(1, -1))
    return best_candidate, float(pred_mean[0]), float(pred_std[0])


def add_boundary_samples_2d(
    n_dims: int, n_samples: int = 8, margin: float = 0.02
) -> np.ndarray:
    """Generate boundary/corner samples for 2D (and fallback for higher dims)."""
    if n_dims != 2:
        return np.array([
            np.random.choice([margin, 1.0 - margin], size=n_dims) for _ in range(n_samples)
        ])
    corners = [
        [margin, margin],
        [margin, 1.0 - margin],
        [1.0 - margin, margin],
        [1.0 - margin, 1.0 - margin],
    ]
    edges = [
        [margin, 0.5],
        [1.0 - margin, 0.5],
        [0.5, margin],
        [0.5, 1.0 - margin],
    ]
    return np.array(corners + edges)
