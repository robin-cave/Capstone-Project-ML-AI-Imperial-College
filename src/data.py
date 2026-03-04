"""
Data loading, function state, and result updates for the BBO capstone.
Works from project root or from notebooks/; paths use PROJECT_ROOT when not overridden.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import ast
import numpy as np

# Project root (parent of src/). Notebooks can run from project root or notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# New layout: data/results/week_1/, week_2/, ... (each with inputs.txt, outputs.txt)
RESULTS_BASE = DATA_DIR / "results"


class FunctionData:
    """Manages data for a single black-box function."""

    def __init__(self, function_id: int, data_dir: Optional[Path] = None):
        self.function_id = function_id
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.function_dir = self.data_dir / f"function_{function_id}"

        # Load initial data
        self.inputs = np.load(self.function_dir / "initial_inputs.npy")
        self.outputs = np.load(self.function_dir / "initial_outputs.npy")

        self.n_samples = len(self.outputs)
        self.n_dims = self.inputs.shape[1]
        self.week_number = 0
        self.history: List[Tuple[int, np.ndarray, float]] = []

    def add_observation(self, x: np.ndarray, y: float, week: Optional[int] = None) -> None:
        if week is None:
            self.week_number += 1
            week = self.week_number
        self.inputs = np.vstack([self.inputs, x.reshape(1, -1)])
        self.outputs = np.append(self.outputs, y)
        self.n_samples += 1
        self.history.append((week, x.copy(), y))

    def get_best(self) -> Tuple[np.ndarray, float]:
        best_idx = np.argmax(self.outputs)
        return self.inputs[best_idx], self.outputs[best_idx]

    def save_weekly_data(self, week: int) -> None:
        week_file_inputs = self.function_dir / f"week_{week}_inputs.npy"
        week_file_outputs = self.function_dir / f"week_{week}_outputs.npy"
        np.save(week_file_inputs, self.inputs)
        np.save(week_file_outputs, self.outputs)

    def get_summary(self) -> Dict:
        return {
            "function_id": self.function_id,
            "n_dims": self.n_dims,
            "n_samples": self.n_samples,
            "best_value": float(np.max(self.outputs)),
            "mean_value": float(np.mean(self.outputs)),
            "std_value": float(np.std(self.outputs)),
        }

    def __repr__(self) -> str:
        best_x, best_y = self.get_best()
        return f"Function {self.function_id}: {self.n_dims}D, {self.n_samples} samples, best={best_y:.6f}"


def _extract_all_lists(file_path: Path) -> List[str]:
    """Extract list structures from a file (one or more [...] blocks)."""
    with open(file_path, "r") as f:
        full_content = f.read()
    list_structures = []
    current_list: List[str] = []
    for line in full_content.split("\n"):
        if line.strip().startswith("["):
            if current_list:
                list_structures.append("\n".join(current_list))
            current_list = [line]
        elif current_list:
            current_list.append(line)
    if current_list:
        list_structures.append("\n".join(current_list))
    return list_structures


def _find_results_dirs() -> Tuple[Optional[Path], bool]:
    """
    Find results directory: prefer data/results/ (new layout with week_1, week_2, ...),
    else legacy 'Week *' in project root.
    Returns (base_path, use_new_layout).
    """
    if RESULTS_BASE.exists():
        week_dirs = sorted(RESULTS_BASE.glob("week_*"))
        if week_dirs and (week_dirs[0] / "inputs.txt").exists():
            return RESULTS_BASE, True
    legacy = sorted(PROJECT_ROOT.glob("Week *"))
    if legacy:
        return legacy[-1], False
    return None, False


def load_results(
    week_index: int = -1,
    results_dir: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """
    Load results by week index.

    Supports two layouts:
    - New: data/results/week_1/, week_2/, ... each with inputs.txt, outputs.txt (one list each).
    - Legacy: single "Week N" directory with inputs.txt/outputs.txt containing multiple lists.

    Args:
        week_index: 0=first week, -1=latest, etc.
        results_dir: Override directory (path to one Week folder or to results base).

    Returns:
        (inputs_dict, outputs_dict, actual_week_number) for functions 1-8.
    """
    if results_dir is not None:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_path}")
        # Single dir with multi-list files (legacy style)
        inputs_file = results_path / "inputs.txt"
        outputs_file = results_path / "outputs.txt"
        if not inputs_file.exists() or not outputs_file.exists():
            raise FileNotFoundError(f"inputs.txt or outputs.txt not in {results_path}")
        inputs_lists = _extract_all_lists(inputs_file)
        outputs_lists = _extract_all_lists(outputs_file)
        if not inputs_lists or not outputs_lists:
            raise ValueError("No valid data found in result files")
        try:
            inputs_content = inputs_lists[week_index]
            outputs_content = outputs_lists[week_index]
            actual_week_num = (
                week_index + 1 if week_index >= 0 else len(inputs_lists) + week_index + 1
            )
        except IndexError:
            raise IndexError(
                f"Week index {week_index} out of range. Available: 0 to {len(inputs_lists) - 1}"
            )
        return _parse_inputs_outputs(inputs_content, outputs_content, results_path, actual_week_num)

    base_path, use_new_layout = _find_results_dirs()
    if base_path is None:
        raise FileNotFoundError("No results directories found (no data/results/week_* or Week *)")

    if use_new_layout:
        week_dirs = sorted(base_path.glob("week_*"))
        if not week_dirs:
            raise FileNotFoundError(f"No week_* dirs in {base_path}")
        try:
            target_dir = week_dirs[week_index]
        except IndexError:
            raise IndexError(
                f"Week index {week_index} out of range. Available: 0 to {len(week_dirs) - 1}"
            )
        inputs_file = target_dir / "inputs.txt"
        outputs_file = target_dir / "outputs.txt"
        if not inputs_file.exists() or not outputs_file.exists():
            raise FileNotFoundError(f"Missing inputs.txt or outputs.txt in {target_dir}")
        inputs_lists = _extract_all_lists(inputs_file)
        outputs_lists = _extract_all_lists(outputs_file)
        if not inputs_lists or not outputs_lists:
            raise ValueError(f"No valid data in {target_dir}")
        # New layout: file in week_n may have one or multiple lists (cumulative); use list at week_index
        if week_index < 0:
            idx = len(inputs_lists) + week_index
        else:
            idx = week_index
        if idx < 0 or idx >= len(inputs_lists):
            raise IndexError(f"Week index {week_index} out of range (file has {len(inputs_lists)} list(s))")
        inputs_content = inputs_lists[idx]
        outputs_content = outputs_lists[idx]
        actual_week_num = week_index + 1 if week_index >= 0 else len(week_dirs) + week_index + 1
        return _parse_inputs_outputs(inputs_content, outputs_content, target_dir, actual_week_num)

    # Legacy: one dir, multi-list files
    inputs_file = base_path / "inputs.txt"
    outputs_file = base_path / "outputs.txt"
    if not inputs_file.exists() or not outputs_file.exists():
        raise FileNotFoundError(f"inputs.txt or outputs.txt not in {base_path}")
    inputs_lists = _extract_all_lists(inputs_file)
    outputs_lists = _extract_all_lists(outputs_file)
    if not inputs_lists or not outputs_lists:
        raise ValueError("No valid data found in result files")
    try:
        inputs_content = inputs_lists[week_index]
        outputs_content = outputs_lists[week_index]
        actual_week_num = week_index + 1 if week_index >= 0 else len(inputs_lists) + week_index + 1
    except IndexError:
        raise IndexError(
            f"Week index {week_index} out of range. Available: 0 to {len(inputs_lists) - 1}"
        )
    return _parse_inputs_outputs(inputs_content, outputs_content, base_path, actual_week_num)


def _parse_inputs_outputs(
    inputs_content: str,
    outputs_content: str,
    location: Path,
    actual_week_num: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Parse input/output text into dicts; print confirmation."""
    inputs_content_clean = re.sub(r"array\(", "(", inputs_content)
    inputs_list = ast.literal_eval(inputs_content_clean)
    outputs_content_clean = re.sub(r"np\.float64\(", "(", outputs_content)
    outputs_list = ast.literal_eval(outputs_content_clean)
    if len(inputs_list) != 8 or len(outputs_list) != 8:
        raise ValueError(f"Expected 8 inputs and 8 outputs, got {len(inputs_list)}, {len(outputs_list)}")
    inputs_dict: Dict[int, np.ndarray] = {}
    outputs_dict: Dict[int, float] = {}
    for func_id in range(1, 9):
        idx = func_id - 1
        inputs_dict[func_id] = (
            inputs_list[idx]
            if isinstance(inputs_list[idx], np.ndarray)
            else np.array(inputs_list[idx])
        )
        ov = outputs_list[idx]
        outputs_dict[func_id] = float(ov.item() if hasattr(ov, "item") else ov)
    print(f"✓ Loaded Week {actual_week_num} from {location}")
    return inputs_dict, outputs_dict, actual_week_num


def initialize_from_history(
    functions_dict: Dict[int, FunctionData],
    week_indices: Optional[List[int]] = None,
    results_dir: Optional[str] = None,
) -> Dict:
    """
    Initialize FunctionData objects with historical weekly results.
    Loads from results and adds observations week by week.
    """
    base_path, use_new = _find_results_dirs()
    if results_dir is not None:
        base_path = Path(results_dir)
        use_new = False
    elif base_path is None:
        raise FileNotFoundError("No results directories found")

    if use_new:
        week_dirs = sorted(base_path.glob("week_*"))
        num_available = len(week_dirs)
    else:
        inp = base_path / "inputs.txt"
        with open(inp, "r") as f:
            content = f.read()
        num_available = content.count("\n[") + (1 if content.strip().startswith("[") else 0)

    if week_indices is None:
        week_indices = list(range(num_available))

    print("=" * 80)
    print("INITIALIZING FUNCTIONS FROM HISTORICAL DATA")
    print("=" * 80)
    print(f"Source: {base_path}")
    print(f"Loading weeks: {[i + 1 for i in week_indices]}")
    print()

    summary: Dict = {}
    for week_idx in week_indices:
        try:
            if results_dir is not None:
                inputs_dict, outputs_dict, week_num = load_results(
                    week_index=week_idx, results_dir=results_dir
                )
            else:
                inputs_dict, outputs_dict, week_num = load_results(week_index=week_idx)
            updated = []
            for func_id in range(1, 9):
                if func_id in functions_dict and func_id in inputs_dict:
                    functions_dict[func_id].add_observation(
                        inputs_dict[func_id], outputs_dict[func_id], week=week_num
                    )
                    updated.append(func_id)
            summary[week_num] = {"num_functions": len(updated), "functions": updated}
            print(f"✓ Week {week_num}: Updated {len(updated)} functions")
        except Exception as e:
            print(f"✗ Week index {week_idx}: {e}")
    print()
    print("=" * 80)
    print(f"✓ Initialization complete: {len(summary)} weeks loaded")
    print("=" * 80)
    print("\nFunction states:")
    for func_id in range(1, 9):
        if func_id in functions_dict:
            f = functions_dict[func_id]
            _, best_y = f.get_best()
            print(f"  Function {func_id} ({f.n_dims}D): {f.n_samples} samples, best={best_y:.6f}")
    print()
    return summary


def initialize_all_weeks(
    functions_dict: Dict[int, FunctionData],
    results_dir: Optional[str] = None,
) -> int:
    """Load all available weeks into functions_dict. Returns number of weeks loaded."""
    summary = initialize_from_history(
        functions_dict=functions_dict,
        week_indices=None,
        results_dir=results_dir,
    )
    return len(summary)


def update_all_functions_with_results(
    functions_dict: Dict[int, FunctionData],
    inputs_dict: Dict[int, np.ndarray],
    outputs_dict: Dict[int, float],
    week: int,
    save: bool = True,
) -> List[Dict]:
    """Update all 8 functions with weekly results; optionally save to disk."""
    print("=" * 80)
    print(f"UPDATING ALL FUNCTIONS WITH WEEK {week} RESULTS")
    print("=" * 80)
    print()
    results: List[Dict] = []
    for func_id in range(1, 9):
        if func_id not in inputs_dict or func_id not in outputs_dict:
            print(f"⚠ Function {func_id} missing from results. Skipping.")
            continue
        x = inputs_dict[func_id]
        y = outputs_dict[func_id]
        func_data = functions_dict[func_id]
        old_best_x, old_best_y = func_data.get_best()
        func_data.add_observation(x, y, week)
        if save:
            func_data.save_weekly_data(week)
        new_best_x, new_best_y = func_data.get_best()
        is_new_best = new_best_y > old_best_y
        improvement = new_best_y - old_best_y if is_new_best else 0.0
        results.append({
            "func_id": func_id,
            "n_dims": func_data.n_dims,
            "y": y,
            "best_y": new_best_y,
            "is_new_best": is_new_best,
            "improvement": improvement,
            "n_samples": func_data.n_samples,
        })
        status = "🌟 NEW BEST!" if is_new_best else "✓"
        print(f"{status} Function {func_id} ({func_data.n_dims}D): y={y:.6f}, best={new_best_y:.6f}")
        if is_new_best:
            print(f"    Improvement: +{improvement:.6f}")
        print(f"    Total samples: {func_data.n_samples}")
        print()
    print("=" * 80)
    print(f"✓ Updated {len(results)} functions. New bests: {sum(1 for r in results if r['is_new_best'])}")
    print("=" * 80)
    return results


def load_latest_results() -> Tuple[Dict[int, np.ndarray], Dict[int, float], int]:
    """Load the most recent week's results."""
    return load_results(week_index=-1)


def load_all_weeks(
    results_dir: Optional[str] = None,
) -> List[Tuple[Dict[int, np.ndarray], Dict[int, float], int]]:
    """Load all available weeks. Returns list of (inputs_dict, outputs_dict, week_num)."""
    base_path, use_new = _find_results_dirs()
    if results_dir is not None:
        base_path = Path(results_dir)
        use_new = False
    if base_path is None:
        raise FileNotFoundError("No results directories found")
    if use_new:
        week_dirs = sorted(base_path.glob("week_*"))
        num_weeks = len(week_dirs)
    else:
        inp = base_path / "inputs.txt"
        with open(inp, "r") as f:
            content = f.read()
        num_weeks = content.count("\n[") + (1 if content.strip().startswith("[") else 0)
    all_results: List[Tuple[Dict[int, np.ndarray], Dict[int, float], int]] = []
    for i in range(num_weeks):
        try:
            if results_dir:
                inp, out, wn = load_results(week_index=i, results_dir=results_dir)
            else:
                inp, out, wn = load_results(week_index=i)
            all_results.append((inp, out, wn))
        except Exception as e:
            print(f"Warning: Could not load week index {i}: {e}")
    print(f"✓ Loaded {len(all_results)} weeks")
    return all_results


def update_function_with_result(
    functions_dict: Dict[int, FunctionData],
    func_id: int,
    x: np.ndarray,
    y: float,
    week: Optional[int] = None,
    save: bool = True,
) -> None:
    """Update a single function with one observation."""
    func_data = functions_dict[func_id]
    func_data.add_observation(x, y, week)
    if save and week is not None:
        func_data.save_weekly_data(week)
    _, best_y = func_data.get_best()
    print(f"✓ Function {func_id}: new y={y:.6f}, best={best_y:.6f}, samples={func_data.n_samples}")
