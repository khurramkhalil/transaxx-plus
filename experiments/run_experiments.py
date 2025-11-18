"""
Main Script for Running All Experiments

This script runs comprehensive experiments comparing all approaches
and generates publication-ready results.

Usage:
    # Quick test (50 batches)
    python experiments/run_experiments.py --data /path/to/cifar10 --quick

    # Full experiments
    python experiments/run_experiments.py --data /path/to/cifar10 --full

    # Custom configuration
    python experiments/run_experiments.py \
        --data /path/to/cifar10 \
        --approaches static threshold stl \
        --scenarios balanced safety_critical \
        --max-batches 100 \
        --output results/
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.utils import cifar10_data_loader, replace_conv_layers, collect_stats, compute_amax
from layers.adapt_convolution_layer import AdaptConv2D
from runtime_monitor.config import MultiplierConfig

from experiments.experiment_runner import ExperimentRunner
from experiments.analysis import ResultAnalyzer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run STL Monitoring Experiments')

    parser.add_argument('--data', type=str, required=True,
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--model', type=str, default='cifar10_repvgg_a0',
                       help='Model from pytorch-cifar-models')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default='./experiment_results',
                       help='Output directory for results')

    # Experiment configuration
    parser.add_argument('--approaches', nargs='+',
                       default=['static', 'threshold', 'stl'],
                       help='Approaches to test')
    parser.add_argument('--scenarios', nargs='+',
                       default=['balanced'],
                       help='STL scenarios (balanced, safety_critical, power_constrained)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Max batches to process (None = all)')

    # Quick/full modes
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (50 batches, fewer scenarios)')
    parser.add_argument('--full', action='store_true',
                       help='Full experiment mode (all batches, all scenarios)')

    # Analysis
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis and plotting')

    return parser.parse_args()


def initialize_model(model_name: str, device: str):
    """Initialize model with approximate layers"""
    print("\n" + "="*70)
    print("Initializing Model")
    print("="*70)

    # Load pretrained model
    print(f"Loading model: {model_name}")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                          model_name, pretrained=True).to(device)

    # Get conv2d layers
    conv_layers = [(name, m) for name, m in model.named_modules()
                   if isinstance(m, torch.nn.Conv2d)
                   and "head" not in name and "reduction" not in name]

    num_layers = len(conv_layers)
    print(f"Found {num_layers} Conv2D layers")

    if num_layers == 0:
        print("Warning: No layers to approximate!")
        return model

    # Initialize with moderate approximation
    half = num_layers // 2
    axx_list = [MultiplierConfig.accurate().to_dict()] * half + \
               [MultiplierConfig.approximate_1l2h().to_dict()] * (num_layers - half)

    print(f"Initializing with {half} accurate + {num_layers - half} approximate layers")
    print("Compiling CUDA extensions (may take a minute)...")

    import time
    start = time.time()
    replace_conv_layers(model, AdaptConv2D, axx_list, 0, 0,
                       layer_count=[0], returned_power=[0], initialize=True)
    print(f"Compilation complete ({time.time() - start:.1f}s)")

    return model


def run_experiments_main(args):
    """Main experiment execution"""
    print("\n" + "="*70)
    print("STL Runtime Monitoring - Experimental Evaluation")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Approaches: {args.approaches}")
    print(f"Scenarios: {args.scenarios}")
    print("="*70)

    # Quick mode configuration
    if args.quick:
        args.max_batches = 50
        args.scenarios = ['balanced']
        print("\n[Quick Mode: 50 batches, balanced scenario only]")

    # Full mode configuration
    if args.full:
        args.max_batches = None
        args.scenarios = ['balanced', 'safety_critical', 'power_constrained']
        print("\n[Full Mode: All batches, all scenarios]")

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    val_data, calib_data = cifar10_data_loader(args.data, batch_size=args.batch_size)
    print(f"Validation batches: {len(val_data)}")
    print(f"Calibration batches: {len(calib_data)}")

    # Initialize model
    model = initialize_model(args.model, args.device)

    # Calibrate
    print("\nCalibrating quantization...")
    with torch.no_grad():
        collect_stats(model, calib_data, num_batches=2, device=args.device)
        compute_amax(model, method="percentile", percentile=99.99, device=args.device)
    print("Calibration complete")

    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output)

    # Run experiments
    print("\n" + "="*70)
    print("Starting Experiments")
    print("="*70)

    results = runner.run_all_experiments(
        model=model,
        data_loader=val_data,
        calib_data=calib_data,
        approaches=args.approaches,
        scenarios=args.scenarios,
        device=args.device,
        max_batches=args.max_batches
    )

    print(f"\n✓ All experiments complete! ({len(results)} total)")

    # Analysis and plotting
    if not args.skip_analysis:
        print("\n" + "="*70)
        print("Generating Analysis and Plots")
        print("="*70)

        analyzer = ResultAnalyzer(args.output)
        analyzer.generate_all_plots()

        print("\n✓ Analysis complete!")

    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print(f"Total experiments: {len(results)}")
    print("\nNext steps:")
    print("  1. Review plots in:", os.path.join(args.output, 'plots'))
    print("  2. Check summary report:", os.path.join(args.output, 'plots/summary_report.txt'))
    print("  3. Use LaTeX table:", os.path.join(args.output, 'plots/comparison_table.tex'))
    print("="*70)


if __name__ == '__main__':
    args = parse_args()

    try:
        run_experiments_main(args)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
