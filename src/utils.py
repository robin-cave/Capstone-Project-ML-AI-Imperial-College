"""
Formatting, visualization, analysis, and progress tracking.
All functions that need function data take an explicit functions_dict argument.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .data import FunctionData, load_results
from .surrogates import SurrogateModel, GPSurrogate
from .acquisition import AcquisitionFunction, optimize_acquisition


def format_for_portal(
    queries: Dict[int, np.ndarray],
    title: str = "FORMATTED QUERIES FOR PORTAL",
) -> None:
    """Format queries for portal submission (hyphen-separated, 6 decimal places)."""
    print("╔" + "=" * 78 + "╗")
    print("║" + f"{title:^78}" + "║")
    print("╠" + "=" * 78 + "╣")
    for func_id in range(1, 9):
        if func_id in queries:
            query = np.clip(queries[func_id], 0, 0.999999)
            query_str = "-".join([f"{x:.6f}" for x in query])
            print(f"║ Function {func_id}: {query_str:<64} ║")
    print("╚" + "=" * 78 + "╝")
    print("\n✓ Copy the formatted queries above and submit to the portal.")


class PredictionTracker:
    """Track predictions vs actual results for surrogate accuracy analysis."""

    def __init__(self) -> None:
        self.predictions: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def record_prediction(
        self,
        week: int,
        func_id: int,
        query: np.ndarray,
        pred_mean: float,
        pred_std: float,
    ) -> None:
        if week not in self.predictions:
            self.predictions[week] = {}
        self.predictions[week][func_id] = {
            "query": query.copy(),
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "actual": None,
            "error": None,
            "within_2std": None,
        }

    def update_actual(self, week: int, func_id: int, actual: float) -> None:
        if week in self.predictions and func_id in self.predictions[week]:
            pred = self.predictions[week][func_id]
            pred["actual"] = actual
            pred["error"] = actual - pred["pred_mean"]
            pred["within_2std"] = abs(pred["error"]) <= 2 * pred["pred_std"]

    def update_all_actuals(self, week: int, outputs_dict: Dict[int, float]) -> None:
        for func_id, actual in outputs_dict.items():
            self.update_actual(week, func_id, actual)

    def analyze_accuracy(self, week: Optional[int] = None) -> None:
        weeks = [week] if week is not None else sorted(self.predictions.keys())
        print("=" * 80)
        print("PREDICTION ACCURACY ANALYSIS")
        print("=" * 80)
        all_errors: List[float] = []
        all_within: List[bool] = []
        for w in weeks:
            if w not in self.predictions:
                continue
            print(f"\nWeek {w}:")
            print(f"{'Func':<6} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Within 2σ':<10}")
            for func_id in range(1, 9):
                if func_id not in self.predictions[w]:
                    continue
                pred = self.predictions[w][func_id]
                if pred["actual"] is None:
                    print(f"{func_id:<6} {pred['pred_mean']:<12.4f} {'pending':<12} {'-':<12} {'-':<10}")
                else:
                    within = "✓" if pred["within_2std"] else "✗"
                    print(f"{func_id:<6} {pred['pred_mean']:<12.4f} {pred['actual']:<12.4f} "
                          f"{pred['error']:<12.4f} {within:<10}")
                    all_errors.append(pred["error"])
                    all_within.append(pred["within_2std"])
        if all_errors:
            err = np.array(all_errors)
            print("\nMean Absolute Error:", np.mean(np.abs(err)))
            print("RMSE:", np.sqrt(np.mean(err ** 2)))
            print(f"Within 2σ: {sum(all_within)}/{len(all_within)}")
        print("=" * 80)


def visualize_2d_surface(
    func_id: int,
    functions_dict: Dict[int, FunctionData],
    surrogate: Optional[SurrogateModel] = None,
    show_acquisition: bool = True,
    acq_func: str = "ucb",
    resolution: int = 50,
    **acq_params: Any,
) -> None:
    """Visualize surrogate mean and uncertainty for 2D functions."""
    func_data = functions_dict[func_id]
    if func_data.n_dims != 2:
        print(f"Function {func_id} is {func_data.n_dims}D, not 2D. Skipping surface plot.")
        return
    if surrogate is None:
        surrogate = GPSurrogate(length_scale=0.5, optimize=True)
        surrogate.fit(func_data.inputs, func_data.outputs)
    x0 = np.linspace(0.02, 0.98, resolution)
    x1 = np.linspace(0.02, 0.98, resolution)
    X0, X1 = np.meshgrid(x0, x1)
    X_grid = np.column_stack([X0.ravel(), X1.ravel()])
    mean, std = surrogate.predict(X_grid)
    mean_grid = mean.reshape(resolution, resolution)
    std_grid = std.reshape(resolution, resolution)
    n_cols = 3 if show_acquisition else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    ax1, ax2 = axes[0], axes[1]
    c1 = ax1.contourf(X0, X1, mean_grid, levels=30, cmap="viridis")
    ax1.scatter(func_data.inputs[:, 0], func_data.inputs[:, 1], c="red", s=100, edgecolors="white", linewidths=2, zorder=5)
    best_x, best_y = func_data.get_best()
    ax1.scatter(best_x[0], best_x[1], c="gold", s=200, marker="*", edgecolors="black", linewidths=2, zorder=6)
    ax1.set_xlabel("x₀")
    ax1.set_ylabel("x₁")
    ax1.set_title(f"Function {func_id}: Mean Prediction")
    plt.colorbar(c1, ax=ax1, label="Predicted Value")
    c2 = ax2.contourf(X0, X1, std_grid, levels=30, cmap="plasma")
    ax2.scatter(func_data.inputs[:, 0], func_data.inputs[:, 1], c="white", s=80, edgecolors="black", linewidths=1)
    ax2.set_xlabel("x₀")
    ax2.set_ylabel("x₁")
    ax2.set_title(f"Function {func_id}: Uncertainty")
    plt.colorbar(c2, ax=ax2, label="Std")
    if show_acquisition:
        if acq_func == "ucb":
            acq_values = AcquisitionFunction.ucb(mean, std, beta=acq_params.get("beta", 2.0))
        elif acq_func == "ei":
            _, y_best = func_data.get_best()
            acq_values = AcquisitionFunction.ei(mean, std, y_best, xi=acq_params.get("xi", 0.01))
        else:
            _, y_best = func_data.get_best()
            acq_values = AcquisitionFunction.pi(mean, std, y_best, xi=acq_params.get("xi", 0.01))
        acq_grid = acq_values.reshape(resolution, resolution)
        ax3 = axes[2]
        ax3.contourf(X0, X1, acq_grid, levels=30, cmap="coolwarm")
        max_idx = np.argmax(acq_values)
        max_x = X_grid[max_idx]
        ax3.scatter(max_x[0], max_x[1], c="yellow", s=200, marker="*", edgecolors="black")
        ax3.set_xlabel("x₀")
        ax3.set_ylabel("x₁")
        ax3.set_title(f"Acquisition ({acq_func.upper()})")
    plt.tight_layout()
    plt.show()


def analyze_function(
    func_id: int,
    functions_dict: Dict[int, FunctionData],
    surrogate: Optional[SurrogateModel] = None,
    acq_func: str = "ucb",
    **acq_params: Any,
) -> tuple:
    """Run full analysis for one function; return (next_query, surrogate)."""
    func_data = functions_dict[func_id]
    if surrogate is None:
        surrogate = GPSurrogate(length_scale=0.5, optimize=True)
    surrogate.fit(func_data.inputs, func_data.outputs)
    next_query = optimize_acquisition(surrogate, func_data, acq_func=acq_func, **acq_params)
    best_x, best_y = func_data.get_best()
    print("=" * 80)
    print(f"FUNCTION {func_id} ANALYSIS")
    print("=" * 80)
    print(f"Dimensions: {func_data.n_dims}D  Samples: {func_data.n_samples}  Best: {best_y:.6f}")
    print(f"Surrogate: {surrogate.get_name()}  Acquisition: {acq_func.upper()}")
    mean, std = surrogate.predict(next_query.reshape(1, -1))
    print(f"Suggested next query: {np.array2string(next_query, precision=6)}")
    print(f"  Predicted: mean={mean[0]:.6f}, std={std[0]:.6f}")
    print("=" * 80)
    if func_data.n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(func_data.inputs[:, 0], func_data.inputs[:, 1], c=func_data.outputs, s=100, cmap="viridis", edgecolors="white")
        ax.scatter(next_query[0], next_query[1], c="red", s=300, marker="*", label="Next Query")
        ax.set_xlabel("x₀")
        ax.set_ylabel("x₁")
        ax.set_title(f"Function {func_id}: Observations and Next Query")
        ax.legend()
        plt.colorbar(ax.collections[0], ax=ax, label="Value")
        plt.show()
    return next_query, surrogate


def analyze_weekly_performance(
    week: int,
    functions_dict: Dict[int, FunctionData],
    inputs_dict: Optional[Dict[int, np.ndarray]] = None,
    outputs_dict: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Analyze performance for a given week. Returns analysis dict."""
    print("=" * 80)
    print(f"ANALYZING WEEK {week} PERFORMANCE")
    print("=" * 80)
    if inputs_dict is None or outputs_dict is None:
        try:
            inputs_dict, outputs_dict, _ = load_results(week_index=week - 1)
        except Exception as e:
            print(f"Could not load results: {e}")
            return {"error": str(e)}
    analysis: Dict[str, Any] = {"week": week, "per_function": {}, "summary": {}, "patterns": []}
    improving: List[int] = []
    stagnant: List[int] = []
    declining: List[int] = []
    print(f"{'Func':<6} {'Dims':<6} {'Old Best':<12} {'New Value':<12} {'Change':<12} {'Status':<15}")
    print("-" * 80)
    for func_id in range(1, 9):
        if func_id not in functions_dict:
            continue
        func_data = functions_dict[func_id]
        old_best = float(np.max(func_data.outputs[:-1])) if func_data.n_samples > 1 else float(func_data.outputs[0])
        new_value = outputs_dict.get(func_id)
        if new_value is None:
            continue
        new_value = float(new_value)
        improvement = new_value - old_best
        pct = (improvement / abs(old_best) * 100) if old_best != 0 else 0
        if improvement > 0.01 * abs(old_best):
            status = "✓ Improving"
            improving.append(func_id)
        elif improvement < -0.01 * abs(old_best):
            status = "✗ Declining"
            declining.append(func_id)
        else:
            status = "→ Stagnant"
            stagnant.append(func_id)
        analysis["per_function"][func_id] = {
            "old_best": old_best,
            "new_value": new_value,
            "improvement": improvement,
            "improvement_pct": pct,
            "status": status,
            "n_samples": func_data.n_samples,
        }
        print(f"{func_id:<6} {func_data.n_dims:<6} {old_best:<12.6f} {new_value:<12.6f} {improvement:+12.6f} {status:<15}")
    improvements_list = [analysis["per_function"][f]["improvement"] for f in analysis["per_function"]]
    analysis["summary"] = {
        "total_functions": len(analysis["per_function"]),
        "improving": len(improving),
        "stagnant": len(stagnant),
        "declining": len(declining),
        "avg_improvement": float(np.mean(improvements_list)) if improvements_list else 0,
        "total_improvement": float(np.sum(improvements_list)) if improvements_list else 0,
    }
    print("=" * 80)
    return analysis


def recommend_strategies(
    analysis: Dict[str, Any],
    current_week: int,
    functions_dict: Dict[int, FunctionData],
    aggressive: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Recommend strategies for next week from analysis output."""
    print("=" * 80)
    print(f"STRATEGY RECOMMENDATIONS FOR WEEK {current_week + 1}")
    print("=" * 80)
    recommendations: Dict[int, Dict[str, Any]] = {}
    for func_id, perf in analysis.get("per_function", {}).items():
        func_data = functions_dict[func_id]
        status = perf["status"]
        strategy: Dict[str, Any] = {"acq_func": "ucb", "beta": 2.0}
        explanation: List[str] = []
        if perf["improvement"] > 0.1 * abs(perf["old_best"]) or "Improving" in status:
            strategy = {"acq_func": "ei", "xi": 0.001} if aggressive else {"acq_func": "ucb", "beta": 1.5}
            explanation.append("Exploitation (improvement)" if not aggressive else "Heavy exploitation")
        elif "Stagnant" in status:
            strategy = {"acq_func": "ucb", "beta": 3.0 if aggressive else 2.5}
            explanation.append("Increased exploration")
        elif "Declining" in status:
            strategy = {"acq_func": "ei", "xi": 0.05}
            explanation.append("Switch to EI")
        if func_data.n_dims >= 6:
            strategy["bound_margin"] = 0.10
            strategy["n_random"] = 2000 + func_data.n_dims * 200
        recommendations[func_id] = {"strategy": strategy, "explanation": "; ".join(explanation) or "Continue"}
        acq_display = strategy["acq_func"].upper()
        if "beta" in strategy:
            acq_display += f" β={strategy['beta']}"
        if "xi" in strategy:
            acq_display += f" ξ={strategy['xi']}"
        print(f"Function {func_id} ({func_data.n_dims}D): {acq_display} — {recommendations[func_id]['explanation']}")
    print("=" * 80)
    return recommendations


def plot_progress(
    functions_dict: Dict[int, FunctionData],
    func_ids: Optional[List[int]] = None,
) -> None:
    """Plot optimization progress (cumulative best) for given functions."""
    if func_ids is None:
        func_ids = list(range(1, 9))
    n_funcs = len(func_ids)
    n_cols = min(3, n_funcs)
    n_rows = (n_funcs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes)
    if n_funcs == 1:
        axes = [axes.flat[0]]
    else:
        axes = axes.flat
    for idx, func_id in enumerate(func_ids):
        func_data = functions_dict[func_id]
        ax = axes[idx]
        cumulative_best = np.maximum.accumulate(func_data.outputs)
        samples = np.arange(1, len(cumulative_best) + 1)
        ax.plot(samples, cumulative_best, "b-", linewidth=2, label="Best")
        ax.scatter(samples, func_data.outputs, c="lightblue", alpha=0.5, s=30)
        if func_data.history:
            n_hist = len(func_data.history)
            hist_vals = [h[2] for h in func_data.history]
            hist_idx = np.arange(len(func_data.outputs) - n_hist, len(func_data.outputs)) + 1
            ax.scatter(hist_idx, hist_vals, c="red", s=100, marker="*", label="Weekly")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.set_title(f"Function {func_id} ({func_data.n_dims}D)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for idx in range(n_funcs, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.show()


def display_competition_summary(functions_dict: Dict[int, FunctionData]) -> None:
    """Print overall competition progress."""
    print("=" * 80)
    print("COMPETITION SUMMARY")
    print("=" * 80)
    if not functions_dict:
        print("No functions loaded.")
        return
    max_week = max(f.week_number for f in functions_dict.values())
    print(f"Total weekly submissions: {max_week}")
    print("Best values by function:")
    print("-" * 80)
    for func_id in range(1, 9):
        if func_id not in functions_dict:
            continue
        func_data = functions_dict[func_id]
        _, best_y = func_data.get_best()
        improvement = best_y - float(func_data.outputs[0]) if len(func_data.outputs) > 0 else 0
        print(f"Function {func_id} ({func_data.n_dims}D): {best_y:.6f} (+{improvement:.6f}, {func_data.n_samples} samples)")
    print("=" * 80)
