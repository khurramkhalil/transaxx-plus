"""
Configuration module for STL runtime monitoring.

Defines all configuration parameters, thresholds, and multiplier specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MultiplierType(Enum):
    """Available approximate multiplier types"""
    ACCURATE = "mul8s_acc"      # Baseline accurate 8-bit multiplier
    APPROX_1L2H = "mul8s_1L2H"  # Low power approximate multiplier
    # Add more multiplier types as needed


@dataclass
class MultiplierConfig:
    """Configuration for a specific multiplier"""
    axx_mult: str
    axx_power: float  # Relative power consumption (1.0 = baseline)
    quant_bits: int = 8
    fake_quant: bool = False

    @classmethod
    def accurate(cls):
        """Factory: Accurate multiplier configuration"""
        return cls(
            axx_mult=MultiplierType.ACCURATE.value,
            axx_power=1.0,
            quant_bits=8,
            fake_quant=False
        )

    @classmethod
    def approximate_1l2h(cls):
        """Factory: 1L2H approximate multiplier configuration"""
        return cls(
            axx_mult=MultiplierType.APPROX_1L2H.value,
            axx_power=0.7082,  # ~30% power savings
            quant_bits=8,
            fake_quant=False
        )

    def to_dict(self):
        """Convert to dictionary for layer replacement"""
        return {
            'axx_mult': self.axx_mult,
            'axx_power': self.axx_power,
            'quant_bits': self.quant_bits,
            'fake_quant': self.fake_quant
        }


@dataclass
class MonitorConfig:
    """Main configuration for STL runtime monitoring"""

    # Signal window configuration
    window_size: int = 100  # Number of recent batches to keep in memory

    # STL formula thresholds
    accuracy_safety_threshold: float = 0.85  # Minimum acceptable accuracy
    accuracy_warning_margin: float = 0.05    # Warning when within this margin
    power_budget_threshold: float = 0.80     # Maximum power (as fraction of baseline)
    quant_drift_threshold: float = 0.10      # Maximum amax drift (relative)

    # Temporal bounds (in number of batches)
    always_window: int = 10          # Check safety over next N batches
    eventually_window: int = 20      # Achievement window for liveness properties
    recovery_time_bound: int = 5     # Max batches to recover from accuracy drop
    recalibration_interval: int = 50 # Batches between forced recalibrations

    # Robustness thresholds
    violation_threshold: float = 0.0     # ρ < 0 means violation
    warning_threshold: float = 0.05      # ρ < 0.05 means warning (predictive)
    safe_margin: float = 0.10            # ρ > 0.10 means safe to increase approximation

    # Adaptation policy
    enable_predictive_adaptation: bool = True   # Use warnings to adapt preemptively
    enable_opportunistic_approximation: bool = True  # Increase approx when safe
    adaptation_cooldown: int = 10  # Min batches between adaptations

    # Monitoring overhead optimization
    enable_lazy_evaluation: bool = True  # Adjust monitoring frequency
    enable_gpu_acceleration: bool = False  # Use GPU for formula evaluation (if available)
    max_monitoring_overhead: float = 0.05  # Target <5% overhead

    # Logging and debugging
    verbose: bool = True
    log_frequency: int = 10  # Log every N batches
    save_traces: bool = True  # Save signal traces for analysis
    trace_output_dir: str = "./runtime_monitor_logs"

    # Multiplier configurations
    multiplier_configs: Dict[str, MultiplierConfig] = field(default_factory=lambda: {
        'safe': MultiplierConfig.accurate(),
        'moderate': MultiplierConfig.approximate_1l2h(),
        'aggressive': MultiplierConfig.approximate_1l2h()  # Can define more aggressive
    })

    # Formula priorities (for conflict resolution)
    formula_priorities: Dict[str, int] = field(default_factory=lambda: {
        'accuracy_safety': 10,        # Highest priority (safety-critical)
        'quant_stability': 9,
        'no_cascading_failures': 8,
        'power_budget': 5,
        'recovery_guarantee': 7,
        'efficiency_goal': 2,
        'convergence': 3,
        'power_latency': 4
    })

    def validate(self):
        """Validate configuration parameters"""
        assert 0.0 < self.accuracy_safety_threshold < 1.0, \
            "Accuracy threshold must be in (0, 1)"
        assert self.window_size > 0, "Window size must be positive"
        assert self.always_window <= self.window_size, \
            "Always window must fit in signal window"
        assert self.warning_threshold > self.violation_threshold, \
            "Warning threshold must be above violation threshold"
        assert self.max_monitoring_overhead > 0.0, \
            "Monitoring overhead must be positive"


# Default configuration instance
DEFAULT_CONFIG = MonitorConfig()


# Predefined scenario configurations
class ScenarioConfigs:
    """Predefined configurations for different deployment scenarios"""

    @staticmethod
    def safety_critical() -> MonitorConfig:
        """Configuration for safety-critical applications (e.g., medical, automotive)"""
        config = MonitorConfig()
        config.accuracy_safety_threshold = 0.92  # Higher safety threshold
        config.accuracy_warning_margin = 0.03     # Earlier warnings
        config.enable_opportunistic_approximation = False  # Conservative
        config.power_budget_threshold = 0.90      # Less aggressive power savings
        return config

    @staticmethod
    def power_constrained() -> MonitorConfig:
        """Configuration for power-constrained embedded systems"""
        config = MonitorConfig()
        config.accuracy_safety_threshold = 0.80  # More tolerance for accuracy loss
        config.power_budget_threshold = 0.60     # Aggressive power savings
        config.enable_opportunistic_approximation = True
        config.safe_margin = 0.08  # More willing to increase approximation
        return config

    @staticmethod
    def balanced() -> MonitorConfig:
        """Balanced configuration (default)"""
        return MonitorConfig()

    @staticmethod
    def debug() -> MonitorConfig:
        """Configuration for debugging and development"""
        config = MonitorConfig()
        config.verbose = True
        config.log_frequency = 1  # Log every batch
        config.save_traces = True
        config.window_size = 50  # Smaller window for faster iteration
        return config
