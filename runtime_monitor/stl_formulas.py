"""
STL Formula Library for Approximate Computing

This module defines all Signal Temporal Logic formulas for monitoring
approximate DNN behavior using the rtamt library.

Categories:
- Safety Properties (Always □)
- Liveness Properties (Eventually ◇)
- Response Properties (Implies →)
- Bounded Response Properties
- Stability Properties
"""

import rtamt
from typing import Dict, List, Optional
from dataclasses import dataclass
from .config import MonitorConfig


@dataclass
class FormulaSpec:
    """Specification for an STL formula"""
    name: str
    description: str
    formula_str: str  # rtamt formula string
    category: str     # 'safety', 'liveness', 'response', 'stability'
    priority: int     # For conflict resolution
    input_signals: List[str]  # Required input signals


class STLFormulaLibrary:
    """
    Library of STL formulas for approximate DNN monitoring.

    Uses rtamt library for STL specification and evaluation.
    Each formula returns a robustness value ρ(φ, σ, t):
        ρ > 0: Formula satisfied with margin ρ
        ρ = 0: Formula marginally satisfied
        ρ < 0: Formula violated by margin |ρ|
    """

    def __init__(self, config: MonitorConfig):
        """
        Initialize formula library with configuration.

        Args:
            config: MonitorConfig with thresholds and bounds
        """
        self.config = config
        self.specs = self._define_formula_specs()
        self.monitors = {}  # Will hold rtamt monitors for each formula
        self._initialize_monitors()

    def _define_formula_specs(self) -> Dict[str, FormulaSpec]:
        """Define all formula specifications"""
        specs = {}

        # ========== SAFETY PROPERTIES (Always) ==========

        # φ₁: Accuracy must always remain above critical threshold
        specs['accuracy_safety'] = FormulaSpec(
            name='accuracy_safety',
            description=f'Accuracy must always be above {self.config.accuracy_safety_threshold}',
            formula_str=f'always[0:{self.config.always_window}](accuracy >= {self.config.accuracy_safety_threshold})',
            category='safety',
            priority=self.config.formula_priorities.get('accuracy_safety', 10),
            input_signals=['accuracy']
        )

        # φ₂: Power consumption must always stay within budget
        specs['power_budget'] = FormulaSpec(
            name='power_budget',
            description=f'Power must always be below {self.config.power_budget_threshold}',
            formula_str=f'always[0:{self.config.always_window}](power <= {self.config.power_budget_threshold})',
            category='safety',
            priority=self.config.formula_priorities.get('power_budget', 5),
            input_signals=['power']
        )

        # φ₃: Quantization drift must never exceed threshold
        specs['quant_stability'] = FormulaSpec(
            name='quant_stability',
            description=f'Quantization drift must stay below {self.config.quant_drift_threshold}',
            formula_str=f'always[0:{self.config.always_window}](amax_drift <= {self.config.quant_drift_threshold})',
            category='stability',
            priority=self.config.formula_priorities.get('quant_stability', 9),
            input_signals=['amax_drift']
        )

        # φ₄: No consecutive batch failures (cascading failures)
        # If accuracy drops below threshold, it must recover within recovery_time_bound
        specs['no_cascading_failures'] = FormulaSpec(
            name='no_cascading_failures',
            description='Accuracy drops must recover quickly',
            formula_str=f'always[0:{self.config.always_window}]((accuracy < {self.config.accuracy_safety_threshold}) implies eventually[0:{self.config.recovery_time_bound}](accuracy >= {self.config.accuracy_safety_threshold + self.config.accuracy_warning_margin}))',
            category='safety',
            priority=self.config.formula_priorities.get('no_cascading_failures', 8),
            input_signals=['accuracy']
        )

        # ========== LIVENESS PROPERTIES (Eventually) ==========

        # φ₅: System should eventually achieve target efficiency
        specs['efficiency_goal'] = FormulaSpec(
            name='efficiency_goal',
            description='Power should eventually drop below target',
            formula_str=f'eventually[0:{self.config.eventually_window}](power <= {self.config.power_budget_threshold * 0.8})',
            category='liveness',
            priority=self.config.formula_priorities.get('efficiency_goal', 2),
            input_signals=['power']
        )

        # ========== BOUNDED RESPONSE PROPERTIES ==========

        # φ₆: If accuracy drops, recover within N batches
        specs['recovery_guarantee'] = FormulaSpec(
            name='recovery_guarantee',
            description=f'Accuracy must recover within {self.config.recovery_time_bound} batches',
            formula_str=f'always[0:{self.config.always_window}]((accuracy < {self.config.accuracy_safety_threshold}) implies eventually[0:{self.config.recovery_time_bound}](accuracy >= {self.config.accuracy_safety_threshold}))',
            category='response',
            priority=self.config.formula_priorities.get('recovery_guarantee', 7),
            input_signals=['accuracy']
        )

        # φ₇: Power savings should manifest quickly after switching to approximate
        specs['power_latency'] = FormulaSpec(
            name='power_latency',
            description='Power reduction should happen quickly',
            formula_str=f'always[0:{self.config.always_window}]((approx_enabled == 1.0) implies eventually[0:5](power < 0.9))',
            category='response',
            priority=self.config.formula_priorities.get('power_latency', 4),
            input_signals=['approx_enabled', 'power']
        )

        # ========== STABILITY PROPERTIES ==========

        # φ₈: Accuracy should not oscillate wildly
        specs['accuracy_stability'] = FormulaSpec(
            name='accuracy_stability',
            description='Accuracy variance should be bounded',
            formula_str=f'always[0:{self.config.always_window}]((accuracy_variance <= 0.05))',
            category='stability',
            priority=6,
            input_signals=['accuracy_variance']
        )

        return specs

    def _initialize_monitors(self):
        """Initialize rtamt monitors for each formula"""
        for name, spec in self.specs.items():
            try:
                # Create discrete-time STL monitor
                monitor = rtamt.STLDiscreteTimeSpecification()
                monitor.name = spec.name

                # Declare input variables
                for signal in spec.input_signals:
                    monitor.declare_var(signal, 'float')

                # Set the specification
                monitor.spec = spec.formula_str

                # Parse the specification
                monitor.parse()

                self.monitors[name] = monitor

            except Exception as e:
                print(f"Warning: Failed to initialize monitor '{name}': {e}")
                print(f"  Formula: {spec.formula_str}")

    def get_monitor(self, formula_name: str) -> Optional[rtamt.STLDiscreteTimeSpecification]:
        """Get rtamt monitor for a specific formula"""
        return self.monitors.get(formula_name)

    def get_all_monitors(self) -> Dict[str, rtamt.STLDiscreteTimeSpecification]:
        """Get all initialized monitors"""
        return self.monitors

    def get_formula_info(self, formula_name: str) -> Optional[FormulaSpec]:
        """Get specification info for a formula"""
        return self.specs.get(formula_name)

    def list_formulas(self) -> List[str]:
        """List all available formula names"""
        return list(self.specs.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List formulas by category"""
        return [name for name, spec in self.specs.items() if spec.category == category]

    def get_required_signals(self) -> set:
        """Get set of all required input signals across all formulas"""
        signals = set()
        for spec in self.specs.values():
            signals.update(spec.input_signals)
        return signals

    def print_summary(self):
        """Print summary of all formulas"""
        print("=" * 70)
        print("STL Formula Library for Approximate Computing")
        print("=" * 70)

        categories = {}
        for spec in self.specs.values():
            if spec.category not in categories:
                categories[spec.category] = []
            categories[spec.category].append(spec)

        for category, specs in sorted(categories.items()):
            print(f"\n{category.upper()} PROPERTIES:")
            print("-" * 70)
            for spec in sorted(specs, key=lambda s: s.priority, reverse=True):
                print(f"  [{spec.priority}] {spec.name}")
                print(f"      {spec.description}")
                print(f"      Formula: {spec.formula_str}")
                print()

        print(f"Total Formulas: {len(self.specs)}")
        print(f"Required Signals: {', '.join(sorted(self.get_required_signals()))}")
        print("=" * 70)


class STLEvaluator:
    """
    Helper class for evaluating STL formulas using rtamt.

    Handles the conversion of signal traces to rtamt format and
    robustness computation.
    """

    @staticmethod
    def prepare_signal_trace(signal_name: str, values: List[float]) -> List[tuple]:
        """
        Prepare signal trace for rtamt evaluation.

        Args:
            signal_name: Name of the signal
            values: List of signal values over time

        Returns:
            List of (time, value) tuples
        """
        return [(t, val) for t, val in enumerate(values)]

    @staticmethod
    def prepare_multi_signal_trace(signals: Dict[str, List[float]]) -> Dict[str, List[tuple]]:
        """
        Prepare multiple signal traces.

        Args:
            signals: Dict mapping signal names to value lists

        Returns:
            Dict mapping signal names to (time, value) tuple lists
        """
        traces = {}
        for signal_name, values in signals.items():
            traces[signal_name] = STLEvaluator.prepare_signal_trace(signal_name, values)
        return traces

    @staticmethod
    def evaluate_formula(monitor: rtamt.STLDiscreteTimeSpecification,
                        signal_traces: Dict[str, List[tuple]]) -> List[tuple]:
        """
        Evaluate STL formula over signal traces.

        Args:
            monitor: rtamt STL monitor
            signal_traces: Dict of signal traces

        Returns:
            List of (time, robustness) tuples
        """
        try:
            # Use online monitoring for incremental evaluation
            robustness = monitor.evaluate(signal_traces)
            return robustness
        except Exception as e:
            print(f"Error evaluating formula: {e}")
            return []

    @staticmethod
    def get_latest_robustness(robustness_trace: List[tuple]) -> float:
        """Extract latest robustness value from trace"""
        if not robustness_trace:
            return 0.0
        return robustness_trace[-1][1]  # (time, robustness) -> robustness

    @staticmethod
    def is_violated(robustness: float) -> bool:
        """Check if formula is violated"""
        return robustness < 0.0

    @staticmethod
    def is_warning(robustness: float, threshold: float = 0.05) -> bool:
        """Check if robustness is in warning zone"""
        return 0.0 <= robustness < threshold

    @staticmethod
    def is_safe(robustness: float, margin: float = 0.10) -> bool:
        """Check if robustness has safe margin"""
        return robustness >= margin
