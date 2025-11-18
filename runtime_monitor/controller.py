"""
Adaptive Approximation Controller

Controls approximation levels based on STL monitoring feedback.
Uses robustness-guided adaptation to balance accuracy and efficiency.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from layers.adapt_convolution_layer import AdaptConv2D
from layers.adapt_linear_layer import AdaPT_Linear
from classification.utils import replace_conv_layers, replace_linear_layers, collect_stats, compute_amax

from .config import MonitorConfig, MultiplierConfig
from .monitor import RuntimeMonitor, MonitoringResult
from .signal_collector import SignalCollector, BatchMetrics


@dataclass
class AdaptationAction:
    """Record of an adaptation action"""
    batch_idx: int
    timestamp: float
    action_type: str  # 'safe_mode', 'reduce', 'increase', 'recalibrate', 'none'
    trigger: str      # What triggered this action
    details: Dict

    def __str__(self):
        return f"Batch {self.batch_idx}: {self.action_type} (trigger: {self.trigger})"


class AdaptiveController:
    """
    Adaptive Approximation Controller using STL-based monitoring.

    Responsibilities:
    - Integrate STL monitor with TransAxx model
    - Execute adaptation actions based on robustness feedback
    - Manage approximation configurations
    - Track adaptation history

    Adaptation strategies:
    1. Safe Mode: Switch all layers to accurate multipliers
    2. Reduce: Decrease approximation in critical layers
    3. Increase: Opportunistically increase approximation
    4. Recalibrate: Update quantization parameters
    """

    def __init__(self,
                 model: torch.nn.Module,
                 monitor: Optional[RuntimeMonitor] = None,
                 config: Optional[MonitorConfig] = None,
                 device: str = 'cuda'):
        """
        Initialize adaptive controller.

        Args:
            model: PyTorch model with approximate layers
            monitor: RuntimeMonitor (creates new if None)
            config: MonitorConfig (uses default if None)
            device: Device for model ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.config = config if config is not None else MonitorConfig()

        # Create or use provided monitor
        if monitor is None:
            self.monitor = RuntimeMonitor(self.config)
        else:
            self.monitor = monitor

        # Get signal collector from monitor
        self.signal_collector = self.monitor.signal_collector

        # Identify approximate layers
        self.conv_layers = self._get_approximate_layers(AdaptConv2D, torch.nn.Conv2d)
        self.linear_layers = self._get_approximate_layers(AdaPT_Linear, torch.nn.Linear)
        self.num_conv_layers = len(self.conv_layers)
        self.num_linear_layers = len(self.linear_layers)

        # Current configuration state
        self.current_config_name = 'moderate'  # Start with moderate approximation
        self.current_conv_config = None
        self.current_linear_config = None

        # Adaptation state
        self.adaptation_history: List[AdaptationAction] = []
        self.last_adaptation_batch = -self.config.adaptation_cooldown
        self.violation_count = 0

        # Performance tracking
        self.total_macs = 0  # Will be computed if needed
        self.total_params = 0

        if self.config.verbose:
            print(f"\nAdaptive Controller Initialized:")
            print(f"  Conv2D layers: {self.num_conv_layers}")
            print(f"  Linear layers: {self.num_linear_layers}")
            print(f"  Starting config: {self.current_config_name}")

    def _get_approximate_layers(self, approx_type, std_type) -> List[Tuple[str, torch.nn.Module]]:
        """Get list of approximate or standard layers that can be approximated"""
        layers = []
        for name, module in self.model.named_modules():
            # Skip head/reduction layers as per TransAxx convention
            if "head" in name or "reduction" in name:
                continue
            if isinstance(module, (approx_type, std_type)):
                layers.append((name, module))
        return layers

    def process_batch(self,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     latency: Optional[float] = None,
                     calib_data=None) -> Dict:
        """
        Main control loop: collect → monitor → adapt.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            latency: Inference latency (optional)
            calib_data: Calibration data for recalibration (optional)

        Returns:
            Dictionary with monitoring results and adaptation actions
        """
        # 1. Collect signals from this batch
        batch_metrics = self.signal_collector.collect_batch_metrics(
            self.model, predictions, targets, latency
        )

        # 2. Monitor STL formulas
        monitoring_result = self.monitor.check_formulas()

        # 3. Decide on adaptation
        action = None
        if monitoring_result.needs_adaptation:
            # Check cooldown period
            if self.signal_collector.batch_count - self.last_adaptation_batch >= self.config.adaptation_cooldown:
                action = self._execute_adaptation(
                    monitoring_result.recommended_action,
                    monitoring_result,
                    calib_data
                )
                self.last_adaptation_batch = self.signal_collector.batch_count

        # 4. Return results
        return {
            'batch_metrics': batch_metrics,
            'monitoring_result': monitoring_result,
            'adaptation_action': action,
            'current_config': self.current_config_name
        }

    def _execute_adaptation(self,
                           action_type: Optional[str],
                           monitoring_result: MonitoringResult,
                           calib_data) -> Optional[AdaptationAction]:
        """
        Execute adaptation action based on monitoring feedback.

        Args:
            action_type: Type of action to execute
            monitoring_result: Current monitoring result
            calib_data: Calibration data (for recalibration)

        Returns:
            AdaptationAction record or None
        """
        if action_type is None or action_type == 'none':
            return None

        trigger = f"violations={monitoring_result.violations}, warnings={monitoring_result.warnings}"

        if action_type == 'safe_mode':
            return self._switch_to_safe_mode(trigger)

        elif action_type == 'reduce':
            return self._reduce_approximation(trigger)

        elif action_type == 'increase':
            return self._increase_approximation(trigger)

        elif action_type == 'recalibrate':
            return self._recalibrate(trigger, calib_data)

        return None

    def _switch_to_safe_mode(self, trigger: str) -> AdaptationAction:
        """
        Emergency: Switch all layers to accurate multipliers.

        This is the highest priority action for safety violations.
        """
        if self.config.verbose:
            print(f"\n[CRITICAL] Switching to SAFE MODE")

        safe_config = self.config.multiplier_configs['safe']

        # Switch all conv layers to accurate
        if self.num_conv_layers > 0:
            axx_list = [safe_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(
                self.model, AdaptConv2D, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        # Switch all linear layers to accurate
        if self.num_linear_layers > 0:
            axx_list = [safe_config.to_dict()] * self.num_linear_layers
            replace_linear_layers(
                self.model, AdaPT_Linear, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        self.current_config_name = 'safe'
        self.violation_count += 1

        action = AdaptationAction(
            batch_idx=self.signal_collector.batch_count,
            timestamp=time.time(),
            action_type='safe_mode',
            trigger=trigger,
            details={'config': 'safe', 'all_layers_accurate': True}
        )

        self.adaptation_history.append(action)
        return action

    def _reduce_approximation(self, trigger: str) -> AdaptationAction:
        """
        Gradual: Reduce approximation in critical layers.

        Strategy: Switch first 50% of layers to accurate, keep rest approximate.
        """
        if self.config.verbose:
            print(f"\n[WARNING] Reducing approximation")

        safe_config = self.config.multiplier_configs['safe']
        moderate_config = self.config.multiplier_configs['moderate']

        # Switch first half of conv layers to accurate
        if self.num_conv_layers > 0:
            half = self.num_conv_layers // 2
            axx_list = [safe_config.to_dict()] * half + \
                      [moderate_config.to_dict()] * (self.num_conv_layers - half)

            replace_conv_layers(
                self.model, AdaptConv2D, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        # Similar for linear layers
        if self.num_linear_layers > 0:
            half = self.num_linear_layers // 2
            axx_list = [safe_config.to_dict()] * half + \
                      [moderate_config.to_dict()] * (self.num_linear_layers - half)

            replace_linear_layers(
                self.model, AdaPT_Linear, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        self.current_config_name = 'reduced'

        action = AdaptationAction(
            batch_idx=self.signal_collector.batch_count,
            timestamp=time.time(),
            action_type='reduce',
            trigger=trigger,
            details={
                'config': 'reduced',
                'accurate_layers': (self.num_conv_layers + self.num_linear_layers) // 2
            }
        )

        self.adaptation_history.append(action)
        return action

    def _increase_approximation(self, trigger: str) -> AdaptationAction:
        """
        Opportunistic: Increase approximation for power savings.

        Only executed when robustness margins are high.
        """
        if self.config.verbose:
            print(f"\n[INFO] Increasing approximation for power savings")

        # Don't increase if already in moderate/aggressive mode
        if self.current_config_name in ['moderate', 'aggressive']:
            return None

        moderate_config = self.config.multiplier_configs['moderate']

        # Switch all layers to moderate approximation
        if self.num_conv_layers > 0:
            axx_list = [moderate_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(
                self.model, AdaptConv2D, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        if self.num_linear_layers > 0:
            axx_list = [moderate_config.to_dict()] * self.num_linear_layers
            replace_linear_layers(
                self.model, AdaPT_Linear, axx_list,
                self.total_macs, self.total_params,
                layer_count=[0], returned_power=[0],
                initialize=False
            )

        self.current_config_name = 'moderate'

        action = AdaptationAction(
            batch_idx=self.signal_collector.batch_count,
            timestamp=time.time(),
            action_type='increase',
            trigger=trigger,
            details={'config': 'moderate'}
        )

        self.adaptation_history.append(action)
        return action

    def _recalibrate(self, trigger: str, calib_data) -> Optional[AdaptationAction]:
        """
        Recalibrate quantization parameters due to drift.

        Args:
            trigger: What triggered recalibration
            calib_data: Calibration dataset

        Returns:
            AdaptationAction or None if calibration data not available
        """
        if calib_data is None:
            if self.config.verbose:
                print("[WARNING] Cannot recalibrate: no calibration data provided")
            return None

        if self.config.verbose:
            print(f"\n[INFO] Recalibrating quantization parameters")

        try:
            with torch.no_grad():
                # Collect statistics
                stats = collect_stats(self.model, calib_data, num_batches=2, device=self.device)

                # Compute new amax values
                amax = compute_amax(self.model, method="percentile",
                                  percentile=99.99, device=self.device)

            # Update calibrated reference in signal collector
            self.signal_collector._store_calibrated_amax(self.model)

            action = AdaptationAction(
                batch_idx=self.signal_collector.batch_count,
                timestamp=time.time(),
                action_type='recalibrate',
                trigger=trigger,
                details={'method': 'percentile', 'percentile': 99.99}
            )

            self.adaptation_history.append(action)
            return action

        except Exception as e:
            if self.config.verbose:
                print(f"[ERROR] Recalibration failed: {e}")
            return None

    def get_adaptation_statistics(self) -> Dict:
        """Get statistics about adaptations"""
        action_counts = {}
        for action in self.adaptation_history:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1

        return {
            'total_adaptations': len(self.adaptation_history),
            'violation_count': self.violation_count,
            'action_counts': action_counts,
            'current_config': self.current_config_name,
            'adaptation_rate': len(self.adaptation_history) / max(1, self.signal_collector.batch_count)
        }

    def print_adaptation_summary(self):
        """Print summary of all adaptations"""
        print("\n" + "="*70)
        print("Adaptation Summary")
        print("="*70)

        stats = self.get_adaptation_statistics()
        print(f"Total Adaptations: {stats['total_adaptations']}")
        print(f"Violation Count: {stats['violation_count']}")
        print(f"Current Config: {stats['current_config']}")
        print(f"Adaptation Rate: {stats['adaptation_rate']:.4f}")

        if stats['action_counts']:
            print("\nAction Breakdown:")
            for action_type, count in sorted(stats['action_counts'].items()):
                print(f"  {action_type}: {count}")

        if self.adaptation_history:
            print("\nAdaptation History:")
            for action in self.adaptation_history[-10:]:  # Last 10
                print(f"  {action}")

        print("="*70)

    def save_adaptation_log(self, filepath: str):
        """Save detailed adaptation log"""
        import json

        log = {
            'statistics': self.get_adaptation_statistics(),
            'adaptation_history': [
                {
                    'batch_idx': a.batch_idx,
                    'timestamp': a.timestamp,
                    'action_type': a.action_type,
                    'trigger': a.trigger,
                    'details': a.details
                }
                for a in self.adaptation_history
            ],
            'layer_info': {
                'num_conv_layers': self.num_conv_layers,
                'num_linear_layers': self.num_linear_layers
            }
        }

        with open(filepath, 'w') as f:
            json.dump(log, f, indent=2)

        if self.config.verbose:
            print(f"Adaptation log saved to: {filepath}")
