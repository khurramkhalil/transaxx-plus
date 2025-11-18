"""
Baseline approaches for comparison with STL monitoring.

Implements three baseline strategies:
1. Static Approximation: Fixed configuration, no adaptation
2. Threshold-Based: Simple if-then rules (reactive only)
3. Oracle: Perfect knowledge of future (upper bound)
"""

import torch
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

from classification.utils import replace_conv_layers, replace_linear_layers, collect_stats, compute_amax
from layers.adapt_convolution_layer import AdaptConv2D
from layers.adapt_linear_layer import AdaPT_Linear
from runtime_monitor.config import MultiplierConfig


@dataclass
class BaselineResult:
    """Result from baseline approach"""
    batch_idx: int
    accuracy: float
    power: float
    num_adaptations: int
    current_config: str
    action_taken: Optional[str] = None


class StaticApproximation:
    """
    Baseline 1: Static Approximation

    Uses fixed multiplier configuration throughout inference.
    No adaptation, no monitoring.
    """

    def __init__(self, model: torch.nn.Module, config_name: str = 'moderate'):
        """
        Args:
            model: PyTorch model with approximate layers
            config_name: 'safe', 'moderate', or 'aggressive'
        """
        self.model = model
        self.config_name = config_name
        self.total_correct = 0
        self.total_samples = 0
        self.batch_count = 0

        # Get layer info
        self.conv_layers = [(name, m) for name, m in model.named_modules()
                           if isinstance(m, (torch.nn.Conv2d, AdaptConv2D))
                           and "head" not in name and "reduction" not in name]
        self.num_conv_layers = len(self.conv_layers)

    def process_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> BaselineResult:
        """Process batch with static configuration"""
        _, predicted = torch.max(predictions, 1)
        num_correct = (predicted == targets).sum().item()
        num_samples = targets.size(0)

        self.total_correct += num_correct
        self.total_samples += num_samples

        accuracy = num_correct / num_samples

        result = BaselineResult(
            batch_idx=self.batch_count,
            accuracy=accuracy,
            power=0.7 if self.config_name == 'moderate' else 1.0,  # Estimate
            num_adaptations=0,
            current_config=self.config_name
        )

        self.batch_count += 1
        return result

    def get_final_accuracy(self) -> float:
        """Get overall accuracy"""
        return self.total_correct / self.total_samples if self.total_samples > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            'approach': 'static',
            'config': self.config_name,
            'final_accuracy': self.get_final_accuracy(),
            'total_batches': self.batch_count,
            'adaptations': 0,
            'avg_power': 0.7 if self.config_name == 'moderate' else 1.0
        }


class ThresholdBased:
    """
    Baseline 2: Threshold-Based Adaptation

    Simple reactive approach:
    - If accuracy < threshold → switch to accurate
    - If accuracy > threshold + margin → switch to approximate

    This is REACTIVE only (acts after violation), unlike STL which is PREDICTIVE.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 accuracy_threshold: float = 0.85,
                 recovery_margin: float = 0.05,
                 cooldown: int = 10):
        """
        Args:
            model: PyTorch model
            accuracy_threshold: Switch to safe if accuracy drops below this
            recovery_margin: Switch back to approx if accuracy exceeds threshold + margin
            cooldown: Minimum batches between adaptations
        """
        self.model = model
        self.accuracy_threshold = accuracy_threshold
        self.recovery_margin = recovery_margin
        self.cooldown = cooldown

        self.current_config = 'moderate'
        self.last_adaptation_batch = -cooldown
        self.total_correct = 0
        self.total_samples = 0
        self.batch_count = 0
        self.adaptation_count = 0
        self.accuracy_history = []

        # Get layer info
        self.conv_layers = [(name, m) for name, m in model.named_modules()
                           if isinstance(m, (torch.nn.Conv2d, AdaptConv2D))
                           and "head" not in name and "reduction" not in name]
        self.num_conv_layers = len(self.conv_layers)

    def process_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> BaselineResult:
        """Process batch with threshold-based adaptation"""
        _, predicted = torch.max(predictions, 1)
        num_correct = (predicted == targets).sum().item()
        num_samples = targets.size(0)

        self.total_correct += num_correct
        self.total_samples += num_samples

        batch_accuracy = num_correct / num_samples
        self.accuracy_history.append(batch_accuracy)

        action_taken = None

        # Check if we can adapt (cooldown period)
        if self.batch_count - self.last_adaptation_batch >= self.cooldown:
            # REACTIVE: Check if accuracy dropped below threshold
            if batch_accuracy < self.accuracy_threshold and self.current_config != 'safe':
                action_taken = self._switch_to_safe()

            # REACTIVE: Check if accuracy recovered enough to go back to approximate
            elif batch_accuracy > self.accuracy_threshold + self.recovery_margin and self.current_config == 'safe':
                action_taken = self._switch_to_moderate()

        result = BaselineResult(
            batch_idx=self.batch_count,
            accuracy=batch_accuracy,
            power=0.7 if self.current_config == 'moderate' else 1.0,
            num_adaptations=self.adaptation_count,
            current_config=self.current_config,
            action_taken=action_taken
        )

        self.batch_count += 1
        return result

    def _switch_to_safe(self) -> str:
        """Switch to safe (accurate) configuration"""
        safe_config = MultiplierConfig.accurate()

        if self.num_conv_layers > 0:
            axx_list = [safe_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(self.model, AdaptConv2D, axx_list, 0, 0,
                              layer_count=[0], returned_power=[0], initialize=False)

        self.current_config = 'safe'
        self.last_adaptation_batch = self.batch_count
        self.adaptation_count += 1
        return 'switch_to_safe'

    def _switch_to_moderate(self) -> str:
        """Switch to moderate (approximate) configuration"""
        moderate_config = MultiplierConfig.approximate_1l2h()

        if self.num_conv_layers > 0:
            axx_list = [moderate_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(self.model, AdaptConv2D, axx_list, 0, 0,
                              layer_count=[0], returned_power=[0], initialize=False)

        self.current_config = 'moderate'
        self.last_adaptation_batch = self.batch_count
        self.adaptation_count += 1
        return 'switch_to_moderate'

    def get_final_accuracy(self) -> float:
        """Get overall accuracy"""
        return self.total_correct / self.total_samples if self.total_samples > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            'approach': 'threshold',
            'threshold': self.accuracy_threshold,
            'final_accuracy': self.get_final_accuracy(),
            'total_batches': self.batch_count,
            'adaptations': self.adaptation_count,
            'current_config': self.current_config,
            'avg_power': sum([0.7 if h > self.accuracy_threshold else 1.0
                            for h in self.accuracy_history]) / len(self.accuracy_history)
        }


class OracleApproach:
    """
    Baseline 3: Oracle (Upper Bound)

    Has perfect knowledge of future accuracy.
    Preemptively switches to safe mode before accuracy drops.

    This represents the BEST POSSIBLE adaptive approach and serves as
    an upper bound for comparison.
    """

    def __init__(self, model: torch.nn.Module, future_window: int = 5):
        """
        Args:
            model: PyTorch model
            future_window: How far ahead the oracle can see
        """
        self.model = model
        self.future_window = future_window

        self.current_config = 'moderate'
        self.total_correct = 0
        self.total_samples = 0
        self.batch_count = 0
        self.adaptation_count = 0
        self.future_accuracy_buffer = []  # Will be filled by experiment runner

        # Get layer info
        self.conv_layers = [(name, m) for name, m in model.named_modules()
                           if isinstance(m, (torch.nn.Conv2d, AdaptConv2D))
                           and "head" not in name and "reduction" not in name]
        self.num_conv_layers = len(self.conv_layers)

    def set_future_accuracy(self, future_accuracy: List[float]):
        """Set oracle's knowledge of future accuracy"""
        self.future_accuracy_buffer = future_accuracy

    def process_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> BaselineResult:
        """Process batch with oracle knowledge"""
        _, predicted = torch.max(predictions, 1)
        num_correct = (predicted == targets).sum().item()
        num_samples = targets.size(0)

        self.total_correct += num_correct
        self.total_samples += num_samples

        batch_accuracy = num_correct / num_samples

        action_taken = None

        # Oracle: Check if accuracy will drop in the future
        if self.batch_count < len(self.future_accuracy_buffer):
            future_start = self.batch_count + 1
            future_end = min(future_start + self.future_window, len(self.future_accuracy_buffer))
            future_accuracies = self.future_accuracy_buffer[future_start:future_end]

            if future_accuracies:
                min_future_acc = min(future_accuracies)

                # Preemptively switch to safe if future accuracy will drop
                if min_future_acc < 0.85 and self.current_config != 'safe':
                    action_taken = self._switch_to_safe()

                # Switch back to approximate if future is safe
                elif min_future_acc > 0.90 and self.current_config == 'safe':
                    action_taken = self._switch_to_moderate()

        result = BaselineResult(
            batch_idx=self.batch_count,
            accuracy=batch_accuracy,
            power=0.7 if self.current_config == 'moderate' else 1.0,
            num_adaptations=self.adaptation_count,
            current_config=self.current_config,
            action_taken=action_taken
        )

        self.batch_count += 1
        return result

    def _switch_to_safe(self) -> str:
        """Switch to safe configuration"""
        safe_config = MultiplierConfig.accurate()

        if self.num_conv_layers > 0:
            axx_list = [safe_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(self.model, AdaptConv2D, axx_list, 0, 0,
                              layer_count=[0], returned_power=[0], initialize=False)

        self.current_config = 'safe'
        self.adaptation_count += 1
        return 'oracle_safe'

    def _switch_to_moderate(self) -> str:
        """Switch to moderate configuration"""
        moderate_config = MultiplierConfig.approximate_1l2h()

        if self.num_conv_layers > 0:
            axx_list = [moderate_config.to_dict()] * self.num_conv_layers
            replace_conv_layers(self.model, AdaptConv2D, axx_list, 0, 0,
                              layer_count=[0], returned_power=[0], initialize=False)

        self.current_config = 'moderate'
        self.adaptation_count += 1
        return 'oracle_moderate'

    def get_final_accuracy(self) -> float:
        """Get overall accuracy"""
        return self.total_correct / self.total_samples if self.total_samples > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            'approach': 'oracle',
            'final_accuracy': self.get_final_accuracy(),
            'total_batches': self.batch_count,
            'adaptations': self.adaptation_count,
            'current_config': self.current_config,
            'avg_power': 0.7  # Estimate
        }
