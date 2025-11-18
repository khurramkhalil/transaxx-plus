"""
Utility functions for runtime monitoring.
"""

import os
import json
import time
from typing import Dict, List, Optional
import numpy as np


def format_robustness(rho: float, threshold_warning: float = 0.05, threshold_safe: float = 0.10) -> str:
    """
    Format robustness value with color indicator.

    Args:
        rho: Robustness value
        threshold_warning: Warning threshold
        threshold_safe: Safe threshold

    Returns:
        Formatted string with indicator
    """
    if rho >= threshold_safe:
        indicator = "✓"  # Safe
    elif rho >= threshold_warning:
        indicator = "~"  # Okay
    elif rho >= 0.0:
        indicator = "⚠"  # Warning
    else:
        indicator = "✗"  # Violation

    return f"{indicator} ρ = {rho:+.4f}"


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values"""
    if not values:
        return {}

    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }


def detect_drift(values: List[float], window: int = 10, threshold: float = 0.1) -> bool:
    """
    Detect drift in signal values.

    Args:
        values: Signal values over time
        window: Window size for comparison
        threshold: Relative drift threshold

    Returns:
        True if drift detected
    """
    if len(values) < 2 * window:
        return False

    recent_mean = np.mean(values[-window:])
    previous_mean = np.mean(values[-2*window:-window])

    if previous_mean == 0:
        return False

    relative_drift = abs(recent_mean - previous_mean) / abs(previous_mean)
    return relative_drift > threshold


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """Compute moving average of signal"""
    if len(values) < window:
        return values

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i+1]))

    return result


def load_monitoring_report(filepath: str) -> Dict:
    """Load monitoring report from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_monitoring_report(filepath: str, report: Dict):
    """Save monitoring report to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)


def merge_reports(report_paths: List[str]) -> Dict:
    """Merge multiple monitoring reports"""
    merged = {
        'robustness_history': {},
        'monitoring_results': [],
        'statistics': {}
    }

    for path in report_paths:
        report = load_monitoring_report(path)

        # Merge robustness histories
        for formula, history in report.get('robustness_history', {}).items():
            if formula not in merged['robustness_history']:
                merged['robustness_history'][formula] = []
            merged['robustness_history'][formula].extend(history)

        # Merge monitoring results
        merged['monitoring_results'].extend(report.get('monitoring_results', []))

    return merged


class Timer:
    """Simple timer for performance measurement"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start timer"""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timer and return elapsed time"""
        if self.start_time is None:
            return 0.0
        self.elapsed = time.time() - self.start_time
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def validate_signal_trace(trace: List[float], signal_name: str, expected_range: Optional[tuple] = None):
    """
    Validate signal trace for common issues.

    Args:
        trace: Signal values
        signal_name: Name of signal
        expected_range: Expected (min, max) range

    Raises:
        ValueError if validation fails
    """
    if not trace:
        raise ValueError(f"Signal '{signal_name}' is empty")

    # Check for NaN/Inf
    if any(np.isnan(v) or np.isinf(v) for v in trace):
        raise ValueError(f"Signal '{signal_name}' contains NaN or Inf values")

    # Check expected range
    if expected_range is not None:
        min_val, max_val = expected_range
        actual_min, actual_max = min(trace), max(trace)
        if actual_min < min_val or actual_max > max_val:
            raise ValueError(
                f"Signal '{signal_name}' out of range: "
                f"expected [{min_val}, {max_val}], got [{actual_min}, {actual_max}]"
            )


def print_formula_summary(formulas: Dict, robustness: Dict[str, float]):
    """
    Print formatted summary of formula evaluations.

    Args:
        formulas: Dictionary of formula specifications
        robustness: Dictionary of robustness values
    """
    print("\nFormula Evaluation Summary:")
    print("=" * 70)

    for name in sorted(robustness.keys()):
        rho = robustness[name]
        formatted = format_robustness(rho)
        print(f"  {name:30s}: {formatted}")

    print("=" * 70)


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create directory for experiment results.

    Args:
        base_dir: Base directory for all experiments
        experiment_name: Name of this experiment

    Returns:
        Path to experiment directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
