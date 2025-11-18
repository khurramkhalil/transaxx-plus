"""
Experiment Runner for STL Monitoring Evaluation

Runs controlled experiments comparing:
1. Static Approximation (no adaptation)
2. Threshold-Based (reactive adaptation)
3. STL Monitoring (predictive adaptation via robustness)
4. Oracle (upper bound, perfect knowledge)

Collects comprehensive metrics for research publication.
"""

import sys
import os
import torch
import time
import json
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_monitor import RuntimeMonitor, AdaptiveController
from runtime_monitor.config import MonitorConfig, ScenarioConfigs
from classification.utils import replace_conv_layers, collect_stats, compute_amax
from layers.adapt_convolution_layer import AdaptConv2D

from .baseline_approaches import StaticApproximation, ThresholdBased, OracleApproach


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    approach: str  # 'static', 'threshold', 'stl', 'oracle'
    model_name: str
    dataset: str
    scenario: Optional[str] = 'balanced'  # For STL: 'safety_critical', 'power_constrained', 'balanced'
    max_batches: Optional[int] = None
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    metrics: Dict
    batch_results: List[Dict]
    adaptations: List[Dict]
    violations: List[Dict]
    warnings: List[Dict]
    timing: Dict


class ExperimentRunner:
    """
    Main experiment runner that orchestrates comparisons.

    Usage:
        runner = ExperimentRunner(output_dir='results/')
        results = runner.run_all_experiments(
            model=model,
            data_loader=val_data,
            calib_data=calib_data
        )
    """

    def __init__(self, output_dir: str = './experiment_results'):
        """
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single_experiment(self,
                             config: ExperimentConfig,
                             model: torch.nn.Module,
                             data_loader,
                             calib_data,
                             device: str = 'cuda') -> ExperimentResult:
        """
        Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            model: PyTorch model (will be copied to avoid interference)
            data_loader: Data loader for evaluation
            calib_data: Calibration data
            device: Device to use

        Returns:
            ExperimentResult with all collected data
        """
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"Approach: {config.approach}")
        print(f"{'='*70}")

        # Create fresh copy of model
        model_copy = copy.deepcopy(model).to(device)
        model_copy.eval()

        # Initialize approach
        if config.approach == 'static':
            approach = StaticApproximation(model_copy, config_name='moderate')
        elif config.approach == 'threshold':
            approach = ThresholdBased(model_copy, accuracy_threshold=0.85)
        elif config.approach == 'oracle':
            approach = OracleApproach(model_copy, future_window=5)
            # Oracle needs future knowledge - we'll do two-pass
            future_accuracy = self._collect_oracle_knowledge(model_copy, data_loader, device)
            approach.set_future_accuracy(future_accuracy)
        elif config.approach == 'stl':
            # STL monitoring approach
            if config.scenario == 'safety_critical':
                monitor_config = ScenarioConfigs.safety_critical()
            elif config.scenario == 'power_constrained':
                monitor_config = ScenarioConfigs.power_constrained()
            else:
                monitor_config = ScenarioConfigs.balanced()

            monitor_config.verbose = False  # Reduce output during experiments
            monitor = RuntimeMonitor(config=monitor_config)
            approach = AdaptiveController(model_copy, monitor, monitor_config, device=device)
        else:
            raise ValueError(f"Unknown approach: {config.approach}")

        # Calibrate if needed
        if config.approach in ['threshold', 'stl', 'oracle']:
            with torch.no_grad():
                collect_stats(model_copy, calib_data, num_batches=2, device=device)
                compute_amax(model_copy, method="percentile", percentile=99.99, device=device)

        # Run inference and collect data
        batch_results = []
        adaptations = []
        violations = []
        warnings = []

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if config.max_batches and batch_idx >= config.max_batches:
                    break

                images, labels = images.to(device), labels.to(device)

                # Inference
                inference_start = time.time()
                outputs = model_copy(images)
                inference_time = time.time() - inference_start

                # Process batch based on approach
                if config.approach in ['static', 'threshold', 'oracle']:
                    result = approach.process_batch(outputs, labels)
                    batch_results.append(asdict(result))

                    if result.action_taken:
                        adaptations.append({
                            'batch': batch_idx,
                            'action': result.action_taken
                        })

                elif config.approach == 'stl':
                    stl_result = approach.process_batch(
                        predictions=outputs,
                        targets=labels,
                        latency=inference_time,
                        calib_data=calib_data
                    )

                    batch_results.append({
                        'batch_idx': batch_idx,
                        'accuracy': stl_result['batch_metrics'].accuracy,
                        'power': stl_result['batch_metrics'].power_estimate,
                        'robustness': stl_result['monitoring_result'].robustness,
                        'current_config': stl_result['current_config']
                    })

                    if stl_result['adaptation_action']:
                        adaptations.append({
                            'batch': batch_idx,
                            'action': stl_result['adaptation_action'].action_type,
                            'trigger': stl_result['adaptation_action'].trigger
                        })

                    if stl_result['monitoring_result'].violations:
                        violations.append({
                            'batch': batch_idx,
                            'formulas': stl_result['monitoring_result'].violations
                        })

                    if stl_result['monitoring_result'].warnings:
                        warnings.append({
                            'batch': batch_idx,
                            'formulas': stl_result['monitoring_result'].warnings
                        })

                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(data_loader)}")

        total_time = time.time() - start_time

        # Collect final metrics
        if config.approach in ['static', 'threshold', 'oracle']:
            final_stats = approach.get_statistics()
        else:  # STL
            final_stats = {
                'approach': 'stl',
                'scenario': config.scenario,
                'final_accuracy': approach.signal_collector.get_running_accuracy(),
                'total_batches': approach.signal_collector.batch_count,
                'adaptations': len(approach.adaptation_history),
                **approach.get_adaptation_statistics()
            }

        timing = {
            'total_time': total_time,
            'avg_time_per_batch': total_time / len(batch_results) if batch_results else 0
        }

        result = ExperimentResult(
            config=config,
            metrics=final_stats,
            batch_results=batch_results,
            adaptations=adaptations,
            violations=violations,
            warnings=warnings,
            timing=timing
        )

        # Save result
        self._save_result(result)

        print(f"\n✓ Experiment Complete:")
        print(f"  Final Accuracy: {final_stats.get('final_accuracy', 0):.4f}")
        print(f"  Adaptations: {len(adaptations)}")
        print(f"  Violations: {len(violations)}")
        print(f"  Time: {total_time:.2f}s")

        return result

    def _collect_oracle_knowledge(self, model, data_loader, device) -> List[float]:
        """First pass: collect true accuracy for oracle"""
        print("  Collecting oracle knowledge (first pass)...")
        accuracies = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                acc = (predicted == labels).float().mean().item()
                accuracies.append(acc)

        return accuracies

    def run_all_experiments(self,
                           model: torch.nn.Module,
                           data_loader,
                           calib_data,
                           approaches: List[str] = ['static', 'threshold', 'stl'],
                           scenarios: List[str] = ['balanced'],
                           device: str = 'cuda',
                           max_batches: Optional[int] = None) -> List[ExperimentResult]:
        """
        Run all experiments (full comparison).

        Args:
            model: Base model to test
            data_loader: Evaluation data
            calib_data: Calibration data
            approaches: List of approaches to test
            scenarios: STL scenarios to test
            device: Device to use
            max_batches: Limit batches for faster testing

        Returns:
            List of ExperimentResult for all experiments
        """
        results = []

        for approach in approaches:
            if approach == 'stl':
                # Run STL with each scenario
                for scenario in scenarios:
                    config = ExperimentConfig(
                        name=f"stl_{scenario}",
                        approach='stl',
                        model_name='model',
                        dataset='dataset',
                        scenario=scenario,
                        max_batches=max_batches
                    )
                    result = self.run_single_experiment(config, model, data_loader, calib_data, device)
                    results.append(result)
            else:
                # Run non-STL approaches
                config = ExperimentConfig(
                    name=approach,
                    approach=approach,
                    model_name='model',
                    dataset='dataset',
                    max_batches=max_batches
                )
                result = self.run_single_experiment(config, model, data_loader, calib_data, device)
                results.append(result)

        # Save summary
        self._save_comparison_summary(results)

        return results

    def _save_result(self, result: ExperimentResult):
        """Save individual experiment result"""
        filepath = os.path.join(self.output_dir, f"{result.config.name}_result.json")

        data = {
            'config': asdict(result.config),
            'metrics': result.metrics,
            'batch_results': result.batch_results,
            'adaptations': result.adaptations,
            'violations': result.violations,
            'warnings': result.warnings,
            'timing': result.timing
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Saved to: {filepath}")

    def _save_comparison_summary(self, results: List[ExperimentResult]):
        """Save comparison summary across all experiments"""
        filepath = os.path.join(self.output_dir, "comparison_summary.json")

        summary = {
            'experiments': [
                {
                    'name': r.config.name,
                    'approach': r.config.approach,
                    'final_accuracy': r.metrics.get('final_accuracy', 0),
                    'total_adaptations': len(r.adaptations),
                    'total_violations': len(r.violations),
                    'total_warnings': len(r.warnings),
                    'total_time': r.timing['total_time']
                }
                for r in results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print("Comparison Summary:")
        print(f"{'='*70}")
        for exp in summary['experiments']:
            print(f"{exp['name']:20s}: Accuracy={exp['final_accuracy']:.4f}, "
                  f"Adaptations={exp['total_adaptations']}, "
                  f"Violations={exp['total_violations']}")
        print(f"{'='*70}")
        print(f"Summary saved to: {filepath}")


def quick_comparison(model, val_data, calib_data, device='cuda', max_batches=50):
    """
    Quick comparison of all approaches (for testing).

    Args:
        model: Model to evaluate
        val_data: Validation data loader
        calib_data: Calibration data loader
        device: Device
        max_batches: Limit batches for speed

    Returns:
        List of experiment results
    """
    runner = ExperimentRunner(output_dir='./quick_comparison_results')

    results = runner.run_all_experiments(
        model=model,
        data_loader=val_data,
        calib_data=calib_data,
        approaches=['static', 'threshold', 'stl'],
        scenarios=['balanced'],
        device=device,
        max_batches=max_batches
    )

    return results
