"""
Runtime STL Monitor

Main monitoring engine that evaluates STL formulas over collected signals
and provides robustness-based feedback for adaptive control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import os

from .config import MonitorConfig
from .stl_formulas import STLFormulaLibrary, STLEvaluator, FormulaSpec
from .signal_collector import SignalCollector


@dataclass
class MonitoringResult:
    """Result of STL monitoring for a single timestep"""
    batch_idx: int
    timestamp: float

    # Robustness values for each formula
    robustness: Dict[str, float] = field(default_factory=dict)

    # Violations and warnings
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safe_formulas: List[str] = field(default_factory=list)

    # Decision recommendation
    needs_adaptation: bool = False
    recommended_action: Optional[str] = None

    # Metadata
    monitoring_latency: float = 0.0  # Time spent monitoring (seconds)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'batch_idx': self.batch_idx,
            'timestamp': self.timestamp,
            'robustness': self.robustness,
            'violations': self.violations,
            'warnings': self.warnings,
            'safe_formulas': self.safe_formulas,
            'needs_adaptation': self.needs_adaptation,
            'recommended_action': self.recommended_action,
            'monitoring_latency': self.monitoring_latency
        }


class RuntimeMonitor:
    """
    Runtime STL Monitor for Approximate DNNs.

    Core responsibilities:
    - Evaluate STL formulas over signal traces using rtamt
    - Compute robustness degrees for each formula
    - Detect violations and warnings
    - Provide adaptation recommendations
    - Log monitoring results

    Uses quantitative STL semantics:
        ρ > 0: Formula satisfied with margin
        ρ = 0: Marginally satisfied
        ρ < 0: Violated
    """

    def __init__(self,
                 config: Optional[MonitorConfig] = None,
                 signal_collector: Optional[SignalCollector] = None):
        """
        Initialize runtime monitor.

        Args:
            config: MonitorConfig (uses default if None)
            signal_collector: SignalCollector (creates new if None)
        """
        self.config = config if config is not None else MonitorConfig()
        self.config.validate()

        # Initialize components
        self.signal_collector = signal_collector if signal_collector is not None \
            else SignalCollector(self.config)
        self.formula_library = STLFormulaLibrary(self.config)
        self.evaluator = STLEvaluator()

        # Monitoring state
        self.monitoring_history: List[MonitoringResult] = []
        self.robustness_history: Dict[str, List[float]] = {
            name: [] for name in self.formula_library.list_formulas()
        }

        # Logging
        self.log_dir = self.config.trace_output_dir
        if self.config.save_traces:
            os.makedirs(self.log_dir, exist_ok=True)

        # Performance tracking
        self.total_monitoring_time = 0.0
        self.monitoring_call_count = 0

        if self.config.verbose:
            print("Runtime Monitor Initialized")
            self.formula_library.print_summary()

    def check_formulas(self) -> MonitoringResult:
        """
        Evaluate all STL formulas on current signal traces.

        Returns:
            MonitoringResult with robustness values and recommendations
        """
        start_time = time.time()

        result = MonitoringResult(
            batch_idx=self.signal_collector.batch_count,
            timestamp=time.time()
        )

        # Get signal traces
        signal_traces = self.signal_collector.get_all_signal_traces()

        # Check if we have enough data
        if not signal_traces or len(signal_traces.get('accuracy', [])) == 0:
            result.monitoring_latency = time.time() - start_time
            return result

        # Prepare traces for rtamt (convert to list of (time, value) tuples)
        rtamt_traces = self.evaluator.prepare_multi_signal_trace(signal_traces)

        # Evaluate each formula
        for formula_name, monitor in self.formula_library.get_all_monitors().items():
            try:
                # Get required signals for this formula
                formula_spec = self.formula_library.get_formula_info(formula_name)
                required_signals = formula_spec.input_signals

                # Check if all required signals are available
                formula_traces = {}
                skip_formula = False
                for signal in required_signals:
                    if signal in rtamt_traces:
                        formula_traces[signal] = rtamt_traces[signal]
                    else:
                        if self.config.verbose:
                            print(f"Warning: Signal '{signal}' not available for formula '{formula_name}'")
                        skip_formula = True
                        break

                if skip_formula:
                    continue

                # Evaluate formula
                robustness_trace = self.evaluator.evaluate_formula(monitor, formula_traces)

                # Get latest robustness value
                if robustness_trace:
                    rho = self.evaluator.get_latest_robustness(robustness_trace)
                    result.robustness[formula_name] = rho

                    # Store in history
                    self.robustness_history[formula_name].append(rho)

                    # Classify result
                    if self.evaluator.is_violated(rho):
                        result.violations.append(formula_name)
                    elif self.evaluator.is_warning(rho, self.config.warning_threshold):
                        result.warnings.append(formula_name)
                    elif self.evaluator.is_safe(rho, self.config.safe_margin):
                        result.safe_formulas.append(formula_name)

            except Exception as e:
                if self.config.verbose:
                    print(f"Error evaluating formula '{formula_name}': {e}")
                continue

        # Determine if adaptation is needed
        result.needs_adaptation = len(result.violations) > 0 or \
                                 (self.config.enable_predictive_adaptation and len(result.warnings) > 0)

        # Generate recommendation
        result.recommended_action = self._generate_recommendation(result)

        # Record monitoring latency
        result.monitoring_latency = time.time() - start_time
        self.total_monitoring_time += result.monitoring_latency
        self.monitoring_call_count += 1

        # Store result
        self.monitoring_history.append(result)

        # Log if configured
        if self.config.verbose and self.signal_collector.batch_count % self.config.log_frequency == 0:
            self._log_result(result)

        return result

    def _generate_recommendation(self, result: MonitoringResult) -> Optional[str]:
        """
        Generate adaptation recommendation based on monitoring result.

        Args:
            result: Current monitoring result

        Returns:
            Recommended action ('safe_mode', 'reduce', 'increase', 'recalibrate', None)
        """
        if not result.robustness:
            return None

        # Priority-based decision making
        priorities = self.config.formula_priorities

        # Critical violations: switch to safe mode
        if result.violations:
            # Find highest priority violation
            highest_priority_violation = max(
                result.violations,
                key=lambda v: priorities.get(v, 0)
            )

            if highest_priority_violation == 'accuracy_safety':
                return 'safe_mode'
            elif highest_priority_violation == 'quant_stability':
                return 'recalibrate'
            elif highest_priority_violation == 'no_cascading_failures':
                return 'safe_mode'
            else:
                return 'reduce'

        # Predictive warnings: reduce approximation
        if self.config.enable_predictive_adaptation and result.warnings:
            if 'accuracy_safety' in result.warnings:
                return 'reduce'
            elif 'quant_stability' in result.warnings:
                return 'recalibrate'

        # Opportunistic: increase approximation if safe
        if self.config.enable_opportunistic_approximation and result.safe_formulas:
            if 'accuracy_safety' in result.safe_formulas and \
               'power_budget' not in result.violations:
                return 'increase'

        return None

    def get_violations(self) -> List[str]:
        """Get currently violated formulas"""
        if self.monitoring_history:
            return self.monitoring_history[-1].violations
        return []

    def get_warnings(self) -> List[str]:
        """Get current warning formulas"""
        if self.monitoring_history:
            return self.monitoring_history[-1].warnings
        return []

    def get_robustness(self, formula_name: str) -> Optional[float]:
        """Get latest robustness value for a formula"""
        if formula_name in self.robustness_history and self.robustness_history[formula_name]:
            return self.robustness_history[formula_name][-1]
        return None

    def get_all_robustness(self) -> Dict[str, float]:
        """Get latest robustness values for all formulas"""
        return {
            name: history[-1] if history else 0.0
            for name, history in self.robustness_history.items()
        }

    def get_monitoring_overhead(self) -> float:
        """
        Get monitoring overhead as fraction of total time.

        Returns:
            Overhead percentage (0.0 to 1.0)
        """
        if self.monitoring_call_count > 0:
            avg_monitoring_time = self.total_monitoring_time / self.monitoring_call_count
            # This is a simplified calculation; in practice you'd track inference time too
            return avg_monitoring_time
        return 0.0

    def _log_result(self, result: MonitoringResult):
        """Log monitoring result to console"""
        print("\n" + "="*70)
        print(f"STL Monitor - Batch {result.batch_idx}")
        print("="*70)

        if result.robustness:
            print("\nRobustness Values:")
            for formula_name, rho in sorted(result.robustness.items(),
                                           key=lambda x: self.config.formula_priorities.get(x[0], 0),
                                           reverse=True):
                status = "✓" if rho >= self.config.warning_threshold else "⚠" if rho >= 0 else "✗"
                print(f"  {status} {formula_name:30s}: ρ = {rho:+.4f}")

        if result.violations:
            print(f"\n❌ VIOLATIONS: {', '.join(result.violations)}")

        if result.warnings:
            print(f"\n⚠️  WARNINGS: {', '.join(result.warnings)}")

        if result.recommended_action:
            print(f"\n💡 RECOMMENDATION: {result.recommended_action}")

        print(f"\nMonitoring Latency: {result.monitoring_latency*1000:.2f} ms")
        print("="*70)

    def save_monitoring_report(self, filepath: Optional[str] = None):
        """Save comprehensive monitoring report"""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"monitor_report_{int(time.time())}.json")

        import json

        report = {
            'config': {
                'window_size': self.config.window_size,
                'accuracy_threshold': self.config.accuracy_safety_threshold,
                'power_threshold': self.config.power_budget_threshold,
            },
            'statistics': self.signal_collector.get_statistics(),
            'robustness_history': {
                name: history for name, history in self.robustness_history.items()
            },
            'monitoring_results': [r.to_dict() for r in self.monitoring_history],
            'performance': {
                'total_monitoring_time': self.total_monitoring_time,
                'avg_monitoring_latency': self.total_monitoring_time / max(1, self.monitoring_call_count),
                'monitoring_call_count': self.monitoring_call_count
            }
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        if self.config.verbose:
            print(f"Monitoring report saved to: {filepath}")

    def plot_robustness_traces(self, save_path: Optional[str] = None):
        """Plot robustness traces over time"""
        try:
            import matplotlib.pyplot as plt

            n_formulas = len(self.robustness_history)
            fig, axes = plt.subplots(n_formulas, 1, figsize=(12, 3*n_formulas))

            if n_formulas == 1:
                axes = [axes]

            for idx, (formula_name, history) in enumerate(self.robustness_history.items()):
                if not history:
                    continue

                ax = axes[idx]
                ax.plot(history, label=formula_name, linewidth=2)
                ax.axhline(y=0, color='r', linestyle='--', label='Violation threshold')
                ax.axhline(y=self.config.warning_threshold, color='orange',
                          linestyle='--', label='Warning threshold')
                ax.axhline(y=self.config.safe_margin, color='g',
                          linestyle='--', label='Safe margin')
                ax.set_xlabel('Batch Index')
                ax.set_ylabel('Robustness')
                ax.set_title(formula_name)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.config.verbose:
                    print(f"Robustness plot saved to: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def reset(self):
        """Reset monitor state"""
        self.monitoring_history.clear()
        for history in self.robustness_history.values():
            history.clear()
        self.total_monitoring_time = 0.0
        self.monitoring_call_count = 0
        self.signal_collector.reset()
