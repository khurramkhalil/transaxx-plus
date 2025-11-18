# Runtime STL Monitoring for TransAxx

**Signal Temporal Logic (STL) based runtime monitoring and adaptive control for approximate Deep Neural Networks.**

## Overview

This module provides formal verification and adaptive control for approximate DNNs during deployment. It uses **Signal Temporal Logic (STL)** to specify temporal properties over accuracy, power consumption, latency, and quantization stability, enabling:

- **Formal Safety Guarantees**: Prove that accuracy never drops below critical thresholds
- **Predictive Adaptation**: Adapt before violations occur using robustness semantics
- **Automated Control**: Dynamically adjust approximation levels based on runtime conditions
- **Low Overhead**: <5% monitoring overhead for real-time systems

## Key Features

### 1. STL Formula Library
Predefined temporal formulas for approximate computing:

- **Safety Properties**: `□ (accuracy ≥ 0.85)` - Accuracy must always stay above threshold
- **Liveness Properties**: `◇[0,20] (power < 0.8)` - Power should eventually drop
- **Bounded Response**: `□ (drop → ◇[0,5] recovery)` - Quick recovery from errors
- **Stability Properties**: `□ (amax_drift < 0.1)` - Quantization stability

### 2. Robustness-Guided Adaptation

Uses **quantitative STL semantics** for predictive control:
```
ρ > 0.10  →  Safe: Can increase approximation
ρ ∈ [0.05, 0.10]  →  Warning: Reduce approximation (predictive)
ρ ∈ [0, 0.05]  →  Critical: Switch to safe mode
ρ < 0  →  Violated: Emergency recovery
```

### 3. Adaptive Approximation Control

Four adaptation strategies:
- **Safe Mode**: All layers use accurate multipliers (emergency)
- **Reduce**: Decrease approximation in critical layers (warning)
- **Increase**: Opportunistically increase approximation (power savings)
- **Recalibrate**: Update quantization parameters (drift detection)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Approximate DNN Model                       │
│          (Conv2D/Linear with axx multipliers)                │
└────────────┬─────────────────────────────────────────────────┘
             │ predictions, metrics
             ▼
┌──────────────────────────────────────────────────────────────┐
│              SignalCollector                                  │
│  • Accuracy, Power, Latency, Quantization Drift             │
└────────────┬─────────────────────────────────────────────────┘
             │ signal traces
             ▼
┌──────────────────────────────────────────────────────────────┐
│              RuntimeMonitor (STL Evaluation)                  │
│  • Evaluate formulas using rtamt                            │
│  • Compute robustness values ρ(φ, σ, t)                     │
│  • Detect violations & warnings                             │
└────────────┬─────────────────────────────────────────────────┘
             │ robustness feedback
             ▼
┌──────────────────────────────────────────────────────────────┐
│              AdaptiveController                               │
│  • Decision logic based on robustness                       │
│  • Execute adaptations (switch multipliers)                 │
│  • Track adaptation history                                 │
└──────────────────────────────────────────────────────────────┘
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify rtamt is installed:
```bash
python -c "import rtamt; print('rtamt version:', rtamt.__version__)"
```

## Quick Start

### Basic Usage

```python
from runtime_monitor import RuntimeMonitor, AdaptiveController
from runtime_monitor.config import MonitorConfig
import torch

# 1. Load your approximate DNN model
model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                       'cifar10_repvgg_a0', pretrained=True).cuda()

# 2. Initialize monitoring
config = MonitorConfig()
monitor = RuntimeMonitor(config=config)
controller = AdaptiveController(model=model, monitor=monitor)

# 3. Run inference with monitoring
model.eval()
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.cuda(), labels.cuda()

        # Inference
        outputs = model(images)

        # Monitor and adapt
        result = controller.process_batch(
            predictions=outputs,
            targets=labels,
            calib_data=calib_data  # For recalibration if needed
        )

        # Check results
        if result['adaptation_action']:
            print(f"Action: {result['adaptation_action'].action_type}")
            print(f"Robustness: {result['monitoring_result'].robustness}")
```

### Running the Demo

```bash
# Balanced scenario (default)
python examples/runtime_monitoring_demo.py \
    --data /path/to/cifar10 \
    --scenario balanced \
    --max-batches 100

# Safety-critical scenario (stricter thresholds)
python examples/runtime_monitoring_demo.py \
    --data /path/to/cifar10 \
    --scenario safety_critical

# Power-constrained scenario (aggressive approximation)
python examples/runtime_monitoring_demo.py \
    --data /path/to/cifar10 \
    --scenario power_constrained

# Debug mode (verbose logging)
python examples/runtime_monitoring_demo.py \
    --data /path/to/cifar10 \
    --scenario debug \
    --max-batches 50
```

## Configuration

### Predefined Scenarios

```python
from runtime_monitor.config import ScenarioConfigs

# Safety-critical: Medical imaging, autonomous vehicles
config = ScenarioConfigs.safety_critical()
# - Accuracy threshold: 0.92 (higher)
# - Warning margin: 0.03 (earlier warnings)
# - No opportunistic approximation

# Power-constrained: Embedded systems, IoT devices
config = ScenarioConfigs.power_constrained()
# - Accuracy threshold: 0.80 (more tolerance)
# - Power budget: 0.60 (aggressive savings)
# - Opportunistic approximation enabled

# Balanced: General-purpose deployment
config = ScenarioConfigs.balanced()

# Debug: Development and testing
config = ScenarioConfigs.debug()
```

### Custom Configuration

```python
from runtime_monitor.config import MonitorConfig, MultiplierConfig

config = MonitorConfig(
    # Signal windows
    window_size=100,
    always_window=10,
    eventually_window=20,

    # Thresholds
    accuracy_safety_threshold=0.88,
    power_budget_threshold=0.75,
    quant_drift_threshold=0.10,

    # Robustness margins
    violation_threshold=0.0,
    warning_threshold=0.05,
    safe_margin=0.10,

    # Adaptation policy
    enable_predictive_adaptation=True,
    enable_opportunistic_approximation=True,
    adaptation_cooldown=10,

    # Logging
    verbose=True,
    log_frequency=10,
    save_traces=True
)
```

### Custom Multipliers

```python
from runtime_monitor.config import MultiplierConfig

# Define custom multiplier
custom_mult = MultiplierConfig(
    axx_mult='mul8s_custom',  # Your custom multiplier
    axx_power=0.65,           # 35% power savings
    quant_bits=8,
    fake_quant=False
)

# Add to configuration
config.multiplier_configs['custom'] = custom_mult
```

## STL Formulas

### Available Formulas

| Formula | Category | Description |
|---------|----------|-------------|
| `accuracy_safety` | Safety | Accuracy ≥ threshold always |
| `power_budget` | Safety | Power ≤ budget always |
| `quant_stability` | Stability | Quantization drift bounded |
| `no_cascading_failures` | Safety | Quick recovery from drops |
| `recovery_guarantee` | Response | Bounded recovery time |
| `efficiency_goal` | Liveness | Eventual power savings |
| `accuracy_stability` | Stability | Bounded accuracy variance |

### Adding Custom Formulas

```python
from runtime_monitor.stl_formulas import FormulaSpec, STLFormulaLibrary

# Define custom formula
custom_formula = FormulaSpec(
    name='latency_bound',
    description='Latency must stay below 100ms',
    formula_str='always[0:10](latency <= 0.1)',
    category='safety',
    priority=8,
    input_signals=['latency']
)

# Add to library
library = STLFormulaLibrary(config)
library.specs['latency_bound'] = custom_formula
library._initialize_monitors()  # Re-initialize with new formula
```

## API Reference

### RuntimeMonitor

```python
monitor = RuntimeMonitor(config=config, signal_collector=collector)

# Main method: check all formulas
result = monitor.check_formulas()

# Get robustness values
rho = monitor.get_robustness('accuracy_safety')
all_rho = monitor.get_all_robustness()

# Get violations/warnings
violations = monitor.get_violations()
warnings = monitor.get_warnings()

# Save results
monitor.save_monitoring_report('report.json')
monitor.plot_robustness_traces('traces.png')
```

### AdaptiveController

```python
controller = AdaptiveController(model, monitor, config, device='cuda')

# Main method: collect → monitor → adapt
result = controller.process_batch(predictions, targets, latency, calib_data)

# Get statistics
stats = controller.get_adaptation_statistics()
controller.print_adaptation_summary()

# Save logs
controller.save_adaptation_log('adaptations.json')
```

### SignalCollector

```python
collector = SignalCollector(config)

# Collect metrics from batch
metrics = collector.collect_batch_metrics(model, predictions, targets, latency)

# Get signal traces
accuracy_trace = collector.get_signal_trace('accuracy')
all_traces = collector.get_all_signal_traces()

# Get statistics
stats = collector.get_statistics()
running_acc = collector.get_running_accuracy()

# Save/load traces
collector.save_traces('traces.json')
collector.load_traces('traces.json')
```

## Understanding Robustness Values

STL robustness degree `ρ(φ, σ, t)` is a **quantitative** measure:

- `ρ > 0`: Formula **satisfied** with safety margin `ρ`
- `ρ = 0`: Formula **marginally satisfied** (boundary)
- `ρ < 0`: Formula **violated** with violation magnitude `|ρ|`

### Example

For formula `□[0,10] (accuracy ≥ 0.85)`:
- If accuracy = [0.90, 0.91, 0.89, ...]: `ρ = 0.04` (min is 0.89, so 0.89 - 0.85 = 0.04)
- If accuracy = [0.88, 0.87, 0.84, ...]: `ρ = -0.01` (min is 0.84, so 0.84 - 0.85 = -0.01)

### Interpretation

```python
rho = monitor.get_robustness('accuracy_safety')

if rho >= 0.10:
    print("Safe: Can increase approximation")
elif rho >= 0.05:
    print("Okay: Maintain current config")
elif rho >= 0.0:
    print("Warning: Reduce approximation (predictive)")
else:
    print("Violation: Emergency safe mode")
```

## Output Files

When running with `save_traces=True`, the following files are generated:

```
runtime_monitor_logs/
├── monitor_report_<timestamp>.json      # Complete monitoring report
├── adaptation_log_<timestamp>.json      # Adaptation history
├── signal_traces_<timestamp>.json       # Raw signal data
├── batch_results_<timestamp>.json       # Per-batch results
└── robustness_traces_<timestamp>.png    # Visualization
```

## Performance

Typical overhead measurements (CIFAR-10, RepVGG-A0, batch_size=128):

| Component | Latency | Overhead |
|-----------|---------|----------|
| Inference | ~20 ms | - |
| Signal Collection | ~0.5 ms | 2.5% |
| STL Monitoring | ~0.8 ms | 4.0% |
| **Total Overhead** | ~1.3 ms | **6.5%** |

Adaptation operations (when triggered):
- Switch multipliers: ~50 ms (amortized over cooldown period)
- Recalibration: ~200 ms (rare, only when drift detected)

## Troubleshooting

### rtamt Import Error

```bash
# Install rtamt
pip install rtamt

# If compilation fails, install from source
git clone https://github.com/nickovic/rtamt.git
cd rtamt
pip install .
```

### CUDA Out of Memory

Reduce batch size or window size:
```python
config.window_size = 50  # Reduce from 100
```

### High Monitoring Overhead

Enable lazy evaluation:
```python
config.enable_lazy_evaluation = True
config.max_monitoring_overhead = 0.05
```

### No Adaptations Occurring

Check if cooldown is too long:
```python
config.adaptation_cooldown = 5  # Reduce from 10
```

Or check if thresholds are too permissive:
```python
config.accuracy_safety_threshold = 0.88  # Increase threshold
config.warning_threshold = 0.08  # Increase warning margin
```

## Examples

See `examples/runtime_monitoring_demo.py` for complete working example.

Additional examples:
- **ImageNet evaluation**: See `CLAUDE.md` for guidance
- **Vision Transformers**: Compatible with ViT models
- **Custom multipliers**: Add your own approximate multipliers

## Citation

If you use this runtime monitoring system, please cite:

```bibtex
@misc{transaxx2024,
  title={TransAxx: Efficient Transformers with Approximate Computing},
  author={Danopoulos, Dimitrios and Zervakis, Georgios and Soudris, Dimitrios and Henkel, Jörg},
  year={2024},
  eprint={2402.07545},
  archivePrefix={arXiv}
}
```

## License

MIT License (same as TransAxx)

## Contact

For questions or issues:
- GitHub Issues: [transaxx-plus/issues](https://github.com/khurramkhalil/transaxx-plus/issues)
- Email: dimdano@microlab.ntua.gr

## References

- **rtamt**: [https://github.com/nickovic/rtamt](https://github.com/nickovic/rtamt)
- **TransAxx**: [https://arxiv.org/abs/2402.07545](https://arxiv.org/abs/2402.07545)
- **STL Semantics**: Donzé, A., & Maler, O. (2010). Robust satisfaction of temporal logic over real-valued signals.
