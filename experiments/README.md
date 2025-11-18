# Experimental Framework for STL Monitoring Evaluation

This directory contains a complete experimental framework for evaluating the STL runtime monitoring system against baseline approaches.

## Overview

The framework compares **four approaches**:

1. **Static Approximation**: Fixed configuration, no adaptation
2. **Threshold-Based**: Simple reactive adaptation (if accuracy < threshold → safe mode)
3. **STL Monitoring**: Predictive adaptation using robustness-guided control (our approach)
4. **Oracle**: Perfect future knowledge (upper bound for comparison)

## Quick Start

### Run Quick Test (50 batches)

```bash
python experiments/run_experiments.py \
    --data /path/to/cifar10 \
    --quick
```

This will:
- Run 3 approaches (static, threshold, STL-balanced)
- Process 50 batches
- Generate all plots and tables
- Save results to `./experiment_results/`

### Run Full Experiments

```bash
python experiments/run_experiments.py \
    --data /path/to/cifar10 \
    --full
```

This will:
- Run all approaches
- Test all STL scenarios (balanced, safety-critical, power-constrained)
- Process entire validation set
- Generate comprehensive analysis

### Custom Configuration

```bash
python experiments/run_experiments.py \
    --data /path/to/cifar10 \
    --approaches static threshold stl \
    --scenarios balanced safety_critical \
    --max-batches 100 \
    --output my_results/
```

## Files

### Core Modules

- **`baseline_approaches.py`**: Baseline implementations
  - `StaticApproximation`: No adaptation
  - `ThresholdBased`: Reactive adaptation
  - `OracleApproach`: Perfect knowledge (upper bound)

- **`experiment_runner.py`**: Experiment orchestration
  - `ExperimentRunner`: Runs experiments and collects data
  - `ExperimentConfig`: Configuration dataclass
  - `ExperimentResult`: Results dataclass

- **`analysis.py`**: Analysis and visualization
  - `ResultAnalyzer`: Generate plots and tables
  - Publication-ready figures
  - LaTeX table generation

- **`run_experiments.py`**: Main entry point
  - Command-line interface
  - Quick/full modes
  - End-to-end execution

## Collected Metrics

For each experiment, we collect:

### Primary Metrics
- **Final Accuracy**: Overall accuracy across all batches
- **Adaptations**: Number of configuration changes
- **Violations**: STL formula violations (safety failures)
- **Warnings**: Early warning signals (predictive)
- **Time**: Total execution time and per-batch latency

### Per-Batch Data
- Batch accuracy
- Power consumption estimate
- Robustness values (for STL)
- Current configuration
- Actions taken

## Generated Outputs

### Plots (in `results/plots/`)

1. **`accuracy_comparison.png`**: Bar chart of final accuracies
2. **`adaptations_comparison.png`**: Number of adaptations
3. **`violations_comparison.png`**: Violations and warnings
4. **`accuracy_over_time.png`**: Accuracy evolution
5. **`accuracy_vs_power.png`**: Pareto curve
6. **`robustness_stl_*.png`**: Robustness traces for each STL scenario

### Tables

- **`comparison_table.tex`**: LaTeX table for publication
- **`summary_report.txt`**: Text summary with key findings

### Raw Data

- **`<approach>_result.json`**: Complete results for each experiment
- **`comparison_summary.json`**: High-level comparison

## Example Workflow

### 1. Run Experiments

```bash
# Quick test to verify everything works
python experiments/run_experiments.py --data /path/to/cifar10 --quick

# Full evaluation
python experiments/run_experiments.py --data /path/to/cifar10 --full
```

### 2. Analyze Results

```python
from experiments.analysis import ResultAnalyzer

# Load and analyze
analyzer = ResultAnalyzer('./experiment_results')

# Generate specific plots
analyzer.plot_accuracy_comparison(save_path='accuracy.png')
analyzer.plot_robustness_traces('stl_balanced', save_path='robustness.png')

# Generate LaTeX table
analyzer.generate_latex_table(save_path='table.tex')

# Get summary
analyzer.generate_summary_report()
```

### 3. Use Results in Paper

Copy from `results/plots/`:
- Figures → Insert in paper
- `comparison_table.tex` → Copy to LaTeX source
- `summary_report.txt` → Reference for writing

## Experimental Design

### Research Questions

**RQ1: Safety Guarantees**
- Does STL detect all violations?
- How early do warnings predict violations?
- Compare violations: static vs. threshold vs. STL

**RQ2: Adaptation Quality**
- Does robustness-guided adaptation outperform threshold-based?
- Measure: accuracy improvement, fewer violations
- Statistical significance testing

**RQ3: Runtime Overhead**
- Measure per-batch latency for each approach
- Confirm <7% overhead for STL monitoring
- Scalability with model size

**RQ4: Scenario Effectiveness**
- Does safety-critical maintain higher accuracy?
- Does power-constrained achieve better power savings?
- Trade-off analysis

### Expected Results

Based on theoretical analysis, we expect:

| Approach | Accuracy | Violations | Adaptations | Overhead |
|----------|----------|------------|-------------|----------|
| Static | Baseline | Many | 0 | 0% |
| Threshold | Similar | Fewer | Moderate | ~1% |
| STL | **Best** | **Fewest** | Optimal | ~6.5% |
| Oracle | Upper bound | 0 | Ideal | N/A |

**Key findings to demonstrate:**
1. STL has **fewer violations** than threshold (predictive vs. reactive)
2. STL maintains **higher accuracy** than static and threshold
3. STL adapts **more intelligently** (not too often, not too rare)
4. Overhead is **acceptable** (<7%) for safety benefits

## Adding New Experiments

### Custom Baseline

```python
from experiments.baseline_approaches import BaselineResult

class MyCustomBaseline:
    def __init__(self, model):
        self.model = model
        # ... initialization

    def process_batch(self, predictions, targets) -> BaselineResult:
        # Your logic here
        return BaselineResult(...)

    def get_statistics(self) -> Dict:
        return {'approach': 'custom', ...}
```

### Custom Analysis

```python
from experiments.analysis import ResultAnalyzer

class MyAnalyzer(ResultAnalyzer):
    def plot_custom_metric(self):
        # Your custom visualization
        pass
```

## Tips for Publication

### For Accuracy Comparison Figure
- Use error bars if running multiple seeds
- Highlight STL as "our approach"
- Show statistical significance (t-test)

### For Robustness Traces
- Show violations as shaded regions
- Annotate adaptation points
- Include warning threshold line

### For Tables
- Bold best results
- Use consistent decimal places (3-4)
- Include standard deviations

### Writing Tips
```markdown
"Our STL-based approach achieves X% higher accuracy than threshold-based
adaptation while reducing violations by Y%. The robustness-guided control
enables *predictive* adaptation, switching to safe mode before violations
occur, unlike reactive threshold-based approaches."
```

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python experiments/run_experiments.py --data /path --batch-size 64
```

### Slow Execution
Use quick mode or limit batches:
```bash
python experiments/run_experiments.py --data /path --max-batches 100
```

### Missing Plots
Install matplotlib:
```bash
pip install matplotlib seaborn
```

## Citation

If you use this experimental framework:

```bibtex
@software{stl_monitoring_experiments,
  title={Experimental Framework for STL Runtime Monitoring},
  author={TransAxx Team},
  year={2025}
}
```
