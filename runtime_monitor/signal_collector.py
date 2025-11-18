"""
Signal Collector for Runtime Monitoring

This module collects runtime signals from approximate DNN inference,
including accuracy, power consumption, latency, and quantization metrics.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from layers.adapt_convolution_layer import AdaptConv2D
from layers.adapt_linear_layer import AdaPT_Linear
from .config import MonitorConfig


@dataclass
class BatchMetrics:
    """Metrics collected from a single inference batch"""
    batch_idx: int
    timestamp: float

    # Accuracy metrics
    accuracy: float  # Top-1 accuracy for this batch
    top5_accuracy: Optional[float] = None
    num_correct: int = 0
    num_samples: int = 0

    # Performance metrics
    inference_latency: float = 0.0  # seconds
    power_estimate: float = 0.0     # relative to baseline

    # Quantization metrics
    amax_drift: float = 0.0         # drift from calibrated values

    # Approximation state
    approx_enabled: float = 0.0     # 1.0 if any approximate layers, 0.0 if all accurate
    num_approx_layers: int = 0

    # Derived metrics
    accuracy_drop: Optional[float] = None  # compared to previous batch

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for signal traces"""
        return {
            'accuracy': self.accuracy,
            'power': self.power_estimate,
            'latency': self.inference_latency,
            'amax_drift': self.amax_drift,
            'approx_enabled': self.approx_enabled,
            'accuracy_variance': 0.0  # Will be computed from history
        }


class SignalCollector:
    """
    Collects and manages runtime signals from approximate DNN inference.

    Responsibilities:
    - Extract metrics from model and inference results
    - Maintain sliding window of historical signals
    - Compute derived signals (variance, drift, etc.)
    - Provide signals in format suitable for STL monitoring
    """

    def __init__(self, config: MonitorConfig):
        """
        Initialize signal collector.

        Args:
            config: MonitorConfig with window size and thresholds
        """
        self.config = config
        self.window_size = config.window_size

        # Signal buffers (sliding windows)
        self.signals = {
            'accuracy': deque(maxlen=self.window_size),
            'power': deque(maxlen=self.window_size),
            'latency': deque(maxlen=self.window_size),
            'amax_drift': deque(maxlen=self.window_size),
            'approx_enabled': deque(maxlen=self.window_size),
            'accuracy_variance': deque(maxlen=self.window_size)
        }

        # Metrics history
        self.batch_history: List[BatchMetrics] = []

        # Calibration reference values
        self.calibrated_amax: Dict[str, float] = {}
        self.baseline_power: float = 1.0

        # Running statistics
        self.batch_count = 0
        self.total_samples = 0
        self.total_correct = 0

    def collect_batch_metrics(self,
                              model: torch.nn.Module,
                              predictions: torch.Tensor,
                              targets: torch.Tensor,
                              latency: Optional[float] = None) -> BatchMetrics:
        """
        Collect metrics from a single inference batch.

        Args:
            model: PyTorch model with approximate layers
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            latency: Inference latency (optional, measured if not provided)

        Returns:
            BatchMetrics object with collected metrics
        """
        metrics = BatchMetrics(
            batch_idx=self.batch_count,
            timestamp=time.time()
        )

        # Compute accuracy
        _, predicted = torch.max(predictions, 1)
        metrics.num_correct = (predicted == targets).sum().item()
        metrics.num_samples = targets.size(0)
        metrics.accuracy = metrics.num_correct / metrics.num_samples

        # Compute top-5 accuracy if predictions have enough classes
        if predictions.size(1) >= 5:
            metrics.top5_accuracy = self._compute_topk_accuracy(predictions, targets, k=5)

        # Estimate power consumption
        metrics.power_estimate = self._estimate_power(model)

        # Measure latency
        metrics.inference_latency = latency if latency is not None else 0.0

        # Check approximation state
        metrics.approx_enabled, metrics.num_approx_layers = self._check_approximation_state(model)

        # Compute quantization drift
        metrics.amax_drift = self._compute_amax_drift(model)

        # Compute accuracy drop (compared to previous batch)
        if len(self.signals['accuracy']) > 0:
            metrics.accuracy_drop = self.signals['accuracy'][-1] - metrics.accuracy

        # Update signals
        self._update_signals(metrics)

        # Update running statistics
        self.batch_count += 1
        self.total_samples += metrics.num_samples
        self.total_correct += metrics.num_correct

        # Store in history
        self.batch_history.append(metrics)

        return metrics

    def _compute_topk_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy"""
        _, topk_pred = predictions.topk(k, dim=1, largest=True, sorted=True)
        correct = topk_pred.eq(targets.view(-1, 1).expand_as(topk_pred))
        return correct.any(dim=1).float().sum().item() / targets.size(0)

    def _estimate_power(self, model: torch.nn.Module) -> float:
        """
        Estimate relative power consumption from approximate layers.

        Returns:
            Power estimate as fraction of baseline (1.0 = accurate baseline)
        """
        total_power = 0.0
        layer_count = 0

        for module in model.modules():
            if isinstance(module, (AdaptConv2D, AdaPT_Linear)):
                if hasattr(module, 'power_percentage'):
                    total_power += module.power_percentage
                    layer_count += 1
                elif hasattr(module, 'axx_mult'):
                    # Estimate based on multiplier type
                    if 'acc' in module.axx_mult:
                        total_power += 1.0  # Accurate = 100% power
                    else:
                        total_power += 0.7  # Approximate ~70% power
                    layer_count += 1

        if layer_count > 0:
            return total_power / (layer_count * 100.0)  # Normalize to [0, 1]
        return 1.0  # Default to baseline

    def _check_approximation_state(self, model: torch.nn.Module) -> Tuple[float, int]:
        """
        Check if model uses approximate multipliers.

        Returns:
            (approx_enabled, num_approx_layers) where approx_enabled is 1.0 if any
            approximate layers, 0.0 if all accurate
        """
        num_approx = 0
        num_total = 0

        for module in model.modules():
            if isinstance(module, (AdaptConv2D, AdaPT_Linear)):
                num_total += 1
                if hasattr(module, 'axx_mult'):
                    if 'acc' not in module.axx_mult:  # Not using accurate multiplier
                        num_approx += 1

        if num_total > 0:
            approx_enabled = 1.0 if num_approx > 0 else 0.0
            return approx_enabled, num_approx

        return 0.0, 0

    def _compute_amax_drift(self, model: torch.nn.Module) -> float:
        """
        Compute drift in quantization amax values from calibrated values.

        Returns:
            Maximum relative drift across all layers
        """
        if not self.calibrated_amax:
            # First pass: store calibrated values
            self._store_calibrated_amax(model)
            return 0.0

        max_drift = 0.0

        for name, module in model.named_modules():
            if isinstance(module, (AdaptConv2D, AdaPT_Linear)):
                # Check if module has amax values
                if hasattr(module, 'amax') and module.amax is not None:
                    current_amax = module.amax.item() if torch.is_tensor(module.amax) else module.amax

                    if name in self.calibrated_amax:
                        calibrated = self.calibrated_amax[name]
                        if calibrated > 0:
                            drift = abs(current_amax - calibrated) / calibrated
                            max_drift = max(max_drift, drift)

        return max_drift

    def _store_calibrated_amax(self, model: torch.nn.Module):
        """Store current amax values as calibration reference"""
        for name, module in model.named_modules():
            if isinstance(module, (AdaptConv2D, AdaPT_Linear)):
                if hasattr(module, 'amax') and module.amax is not None:
                    amax_val = module.amax.item() if torch.is_tensor(module.amax) else module.amax
                    self.calibrated_amax[name] = amax_val

    def _update_signals(self, metrics: BatchMetrics):
        """Update signal buffers with new metrics"""
        signal_dict = metrics.to_dict()

        for signal_name, value in signal_dict.items():
            if signal_name in self.signals:
                self.signals[signal_name].append(value)

        # Compute and update accuracy variance
        if len(self.signals['accuracy']) >= 2:
            variance = np.var(list(self.signals['accuracy']))
            self.signals['accuracy_variance'][-1] = variance
        else:
            self.signals['accuracy_variance'].append(0.0)

    def get_signal_trace(self, signal_name: str) -> List[float]:
        """Get complete trace for a signal"""
        if signal_name in self.signals:
            return list(self.signals[signal_name])
        return []

    def get_all_signal_traces(self) -> Dict[str, List[float]]:
        """Get all signal traces as dictionary"""
        return {name: list(values) for name, values in self.signals.items()}

    def get_latest_value(self, signal_name: str) -> Optional[float]:
        """Get latest value for a signal"""
        if signal_name in self.signals and len(self.signals[signal_name]) > 0:
            return self.signals[signal_name][-1]
        return None

    def get_running_accuracy(self) -> float:
        """Get overall running accuracy across all batches"""
        if self.total_samples > 0:
            return self.total_correct / self.total_samples
        return 0.0

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics"""
        stats = {
            'batch_count': self.batch_count,
            'running_accuracy': self.get_running_accuracy(),
            'total_samples': self.total_samples,
        }

        # Add mean/std for each signal
        for signal_name, values in self.signals.items():
            if len(values) > 0:
                stats[f'{signal_name}_mean'] = np.mean(values)
                stats[f'{signal_name}_std'] = np.std(values)
                stats[f'{signal_name}_min'] = np.min(values)
                stats[f'{signal_name}_max'] = np.max(values)

        return stats

    def reset(self):
        """Reset all signals and statistics"""
        for signal in self.signals.values():
            signal.clear()

        self.batch_history.clear()
        self.calibrated_amax.clear()
        self.batch_count = 0
        self.total_samples = 0
        self.total_correct = 0

    def save_traces(self, filepath: str):
        """Save signal traces to file"""
        import json

        data = {
            'signals': self.get_all_signal_traces(),
            'statistics': self.get_statistics(),
            'config': {
                'window_size': self.window_size,
                'batch_count': self.batch_count
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_traces(self, filepath: str):
        """Load signal traces from file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore signals
        for signal_name, values in data['signals'].items():
            if signal_name in self.signals:
                self.signals[signal_name].clear()
                self.signals[signal_name].extend(values)

        self.batch_count = data['config']['batch_count']
