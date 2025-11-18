"""
Unit tests for runtime monitoring system.

Run with: pytest tests/test_runtime_monitor.py -v
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_monitor import RuntimeMonitor, SignalCollector, AdaptiveController
from runtime_monitor.config import MonitorConfig, MultiplierConfig
from runtime_monitor.stl_formulas import STLFormulaLibrary, STLEvaluator


def test_monitor_config():
    """Test monitor configuration"""
    print("\n=== Testing MonitorConfig ===")

    config = MonitorConfig()
    config.validate()

    assert config.window_size > 0
    assert 0 < config.accuracy_safety_threshold < 1.0
    assert config.warning_threshold > config.violation_threshold

    print("✓ MonitorConfig validation passed")


def test_multiplier_config():
    """Test multiplier configuration"""
    print("\n=== Testing MultiplierConfig ===")

    accurate = MultiplierConfig.accurate()
    assert accurate.axx_mult == 'mul8s_acc'
    assert accurate.axx_power == 1.0

    approx = MultiplierConfig.approximate_1l2h()
    assert approx.axx_mult == 'mul8s_1L2H'
    assert approx.axx_power < 1.0

    print("✓ MultiplierConfig passed")


def test_stl_formula_library():
    """Test STL formula library initialization"""
    print("\n=== Testing STLFormulaLibrary ===")

    config = MonitorConfig()
    library = STLFormulaLibrary(config)

    # Check formulas exist
    formulas = library.list_formulas()
    assert len(formulas) > 0
    print(f"  Found {len(formulas)} formulas")

    # Check monitors initialized
    monitors = library.get_all_monitors()
    assert len(monitors) > 0
    print(f"  Initialized {len(monitors)} monitors")

    # Check required signals
    required_signals = library.get_required_signals()
    print(f"  Required signals: {required_signals}")

    assert 'accuracy' in required_signals
    assert 'power' in required_signals

    print("✓ STLFormulaLibrary passed")


def test_signal_collector():
    """Test signal collector"""
    print("\n=== Testing SignalCollector ===")

    config = MonitorConfig()
    collector = SignalCollector(config)

    # Simulate model and predictions
    class DummyModule:
        def __init__(self):
            self.amax = torch.tensor(1.0)
            self.axx_mult = 'mul8s_acc'

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = DummyModule()

        def modules(self):
            return [self.dummy]

    model = DummyModel()

    # Create dummy predictions and targets
    predictions = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    # Collect metrics
    metrics = collector.collect_batch_metrics(model, predictions, targets, latency=0.01)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.num_samples == 32
    assert metrics.inference_latency == 0.01

    print(f"  Batch accuracy: {metrics.accuracy:.4f}")
    print(f"  Power estimate: {metrics.power_estimate:.4f}")

    # Test signal traces
    traces = collector.get_all_signal_traces()
    assert 'accuracy' in traces
    assert len(traces['accuracy']) == 1

    print("✓ SignalCollector passed")


def test_stl_evaluator():
    """Test STL evaluator"""
    print("\n=== Testing STLEvaluator ===")

    evaluator = STLEvaluator()

    # Test signal preparation
    signal_values = [0.9, 0.91, 0.89, 0.90, 0.92]
    trace = evaluator.prepare_signal_trace('accuracy', signal_values)

    assert len(trace) == len(signal_values)
    assert trace[0] == (0, 0.9)
    assert trace[-1] == (4, 0.92)

    print(f"  Prepared trace: {len(trace)} points")
    print("✓ STLEvaluator passed")


def test_runtime_monitor():
    """Test runtime monitor with synthetic signals"""
    print("\n=== Testing RuntimeMonitor ===")

    config = MonitorConfig()
    config.verbose = False  # Reduce output during test

    monitor = RuntimeMonitor(config)

    # Manually inject synthetic signal data
    # Simulate 20 batches with varying accuracy
    for i in range(20):
        # Accuracy that stays above threshold
        accuracy = 0.85 + 0.05 * np.sin(i * 0.3)
        monitor.signal_collector.signals['accuracy'].append(accuracy)
        monitor.signal_collector.signals['power'].append(0.75)
        monitor.signal_collector.signals['latency'].append(0.02)
        monitor.signal_collector.signals['amax_drift'].append(0.05)
        monitor.signal_collector.signals['approx_enabled'].append(1.0)
        monitor.signal_collector.signals['accuracy_variance'].append(0.01)

    monitor.signal_collector.batch_count = 20

    # Check formulas
    result = monitor.check_formulas()

    assert result.batch_idx == 20
    assert len(result.robustness) > 0

    print(f"  Evaluated {len(result.robustness)} formulas")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Warnings: {len(result.warnings)}")

    # Print robustness values
    for formula, rho in result.robustness.items():
        print(f"    {formula}: ρ = {rho:+.4f}")

    print("✓ RuntimeMonitor passed")


def test_adaptive_controller_init():
    """Test adaptive controller initialization"""
    print("\n=== Testing AdaptiveController ===")

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(16*30*30, 10)
    )

    config = MonitorConfig()
    config.verbose = False

    controller = AdaptiveController(model=model, config=config, device='cpu')

    assert controller.model is model
    assert controller.monitor is not None
    assert controller.signal_collector is not None

    print(f"  Conv layers: {controller.num_conv_layers}")
    print(f"  Linear layers: {controller.num_linear_layers}")

    print("✓ AdaptiveController initialization passed")


def test_end_to_end_synthetic():
    """End-to-end test with synthetic data"""
    print("\n=== Testing End-to-End with Synthetic Data ===")

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    model.eval()

    config = MonitorConfig()
    config.verbose = False
    config.enable_predictive_adaptation = False  # Simplify for test

    monitor = RuntimeMonitor(config)
    controller = AdaptiveController(model, monitor, config, device='cpu')

    # Simulate batches
    num_batches = 10
    for i in range(num_batches):
        # Generate synthetic data
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 5, (32,))

        with torch.no_grad():
            outputs = model(inputs)

        # Process batch
        result = controller.process_batch(
            predictions=outputs,
            targets=targets,
            latency=0.01,
            calib_data=None
        )

        assert result['batch_metrics'] is not None
        assert result['monitoring_result'] is not None

        if i == 0:
            print(f"  Batch {i}: Accuracy = {result['batch_metrics'].accuracy:.4f}")

    # Get statistics
    stats = controller.get_adaptation_statistics()
    print(f"  Total batches: {num_batches}")
    print(f"  Total adaptations: {stats['total_adaptations']}")
    print(f"  Current config: {stats['current_config']}")

    print("✓ End-to-end test passed")


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("Runtime Monitor Test Suite")
    print("="*70)

    tests = [
        test_monitor_config,
        test_multiplier_config,
        test_stl_formula_library,
        test_signal_collector,
        test_stl_evaluator,
        test_runtime_monitor,
        test_adaptive_controller_init,
        test_end_to_end_synthetic
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
