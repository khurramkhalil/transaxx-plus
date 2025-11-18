"""
Runtime STL Monitoring Demo for TransAxx

This script demonstrates complete STL-based runtime monitoring for
approximate DNNs using the TransAxx framework.

Features:
- Load pretrained model (CIFAR-10)
- Initialize STL monitoring with custom formulas
- Run inference with real-time monitoring
- Adaptive approximation based on robustness feedback
- Visualization and reporting

Usage:
    python examples/runtime_monitoring_demo.py --data /path/to/cifar10 --scenario balanced
"""

import sys
import os
import argparse
import time
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_monitor import RuntimeMonitor, AdaptiveController, MonitorConfig, ScenarioConfigs
from runtime_monitor.stl_formulas import STLFormulaLibrary
from classification.utils import cifar10_data_loader, replace_conv_layers
from layers.adapt_convolution_layer import AdaptConv2D


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Runtime STL Monitoring Demo')

    parser.add_argument('--data', type=str, default='/workspace/datasets/cifar10_data',
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--scenario', type=str, default='balanced',
                       choices=['balanced', 'safety_critical', 'power_constrained', 'debug'],
                       help='Monitoring scenario configuration')
    parser.add_argument('--model', type=str, default='cifar10_repvgg_a0',
                       help='Model name from pytorch-cifar-models')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process (None = all)')
    parser.add_argument('--save-dir', type=str, default='./runtime_monitor_results',
                       help='Directory to save results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    return parser.parse_args()


def setup_monitoring(scenario: str, save_dir: str):
    """
    Setup monitoring configuration based on scenario.

    Args:
        scenario: Scenario name
        save_dir: Directory for saving logs

    Returns:
        MonitorConfig
    """
    if scenario == 'safety_critical':
        config = ScenarioConfigs.safety_critical()
    elif scenario == 'power_constrained':
        config = ScenarioConfigs.power_constrained()
    elif scenario == 'debug':
        config = ScenarioConfigs.debug()
    else:
        config = ScenarioConfigs.balanced()

    config.trace_output_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    return config


def initialize_model_with_approximation(model, device):
    """
    Initialize model with approximate layers.

    Args:
        model: PyTorch model
        device: Device to use

    Returns:
        Modified model with approximate layers
    """
    print("\n" + "="*70)
    print("Initializing Model with Approximate Layers")
    print("="*70)

    # Get conv2d layers to approximate
    conv2d_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Conv2d)
                     and "head" not in name and "reduction" not in name]

    num_layers = len(conv2d_layers)
    print(f"Found {num_layers} Conv2D layers to approximate")

    if num_layers == 0:
        print("Warning: No layers found to approximate")
        return model

    # Initialize with moderate approximation (mix of accurate and approximate)
    # Start conservative: first 50% accurate, rest approximate
    from runtime_monitor.config import MultiplierConfig

    half = num_layers // 2
    axx_list = [MultiplierConfig.accurate().to_dict()] * half + \
               [MultiplierConfig.approximate_1l2h().to_dict()] * (num_layers - half)

    print(f"Configuration: {half} accurate + {num_layers - half} approximate layers")

    # Replace layers (initialize=True for first time compilation)
    print("Compiling CUDA extensions (this may take a minute)...")
    start = time.time()
    replace_conv_layers(model, AdaptConv2D, axx_list,
                       0, 0, layer_count=[0], returned_power=[0],
                       initialize=True)
    print(f"Compilation complete in {time.time() - start:.1f} seconds")

    return model


def run_calibration(model, calib_data, device):
    """
    Run quantization calibration.

    Args:
        model: Model to calibrate
        calib_data: Calibration dataset
        device: Device to use
    """
    print("\n" + "="*70)
    print("Running Quantization Calibration")
    print("="*70)

    from classification.utils import collect_stats, compute_amax

    with torch.no_grad():
        print("Collecting statistics on calibration data...")
        stats = collect_stats(model, calib_data, num_batches=2, device=device)

        print("Computing amax values (percentile method)...")
        amax = compute_amax(model, method="percentile", percentile=99.99, device=device)

    print("Calibration complete")


def run_monitored_inference(model, val_data, calib_data, controller, args):
    """
    Run inference with STL monitoring and adaptive control.

    Args:
        model: Model to evaluate
        val_data: Validation data loader
        calib_data: Calibration data loader
        controller: AdaptiveController
        args: Command line arguments

    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print("Starting Monitored Inference")
    print("="*70)

    model.eval()
    results = {
        'batch_results': [],
        'final_accuracy': 0.0,
        'adaptations': [],
        'monitoring_overhead': 0.0
    }

    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_data):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            images, labels = images.to(args.device), labels.to(args.device)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time

            # Process batch with monitoring and adaptation
            result = controller.process_batch(
                predictions=outputs,
                targets=labels,
                latency=inference_time,
                calib_data=calib_data
            )

            # Track results
            batch_metrics = result['batch_metrics']
            total_correct += batch_metrics.num_correct
            total_samples += batch_metrics.num_samples
            batch_count += 1

            # Store batch result
            results['batch_results'].append({
                'batch_idx': batch_idx,
                'accuracy': batch_metrics.accuracy,
                'robustness': result['monitoring_result'].robustness,
                'violations': result['monitoring_result'].violations,
                'warnings': result['monitoring_result'].warnings,
                'action': result['adaptation_action'].action_type if result['adaptation_action'] else 'none',
                'config': result['current_config']
            })

            # Record adaptations
            if result['adaptation_action']:
                results['adaptations'].append(result['adaptation_action'])

            # Periodic progress
            if (batch_idx + 1) % 10 == 0:
                running_acc = total_correct / total_samples
                print(f"Batch {batch_idx + 1}/{len(val_data)}: "
                      f"Running Accuracy = {running_acc:.4f}, "
                      f"Config = {result['current_config']}")

    # Final statistics
    results['final_accuracy'] = total_correct / total_samples
    results['monitoring_overhead'] = controller.monitor.get_monitoring_overhead()

    print("\n" + "="*70)
    print("Inference Complete")
    print("="*70)
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Total Batches: {batch_count}")
    print(f"Total Adaptations: {len(results['adaptations'])}")
    print(f"Monitoring Overhead: {results['monitoring_overhead']*1000:.2f} ms/batch")

    return results


def save_results(controller, results, args):
    """Save all results and generate reports"""
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    save_dir = args.save_dir
    timestamp = int(time.time())

    # Save monitoring report
    monitor_report_path = os.path.join(save_dir, f'monitor_report_{timestamp}.json')
    controller.monitor.save_monitoring_report(monitor_report_path)

    # Save adaptation log
    adaptation_log_path = os.path.join(save_dir, f'adaptation_log_{timestamp}.json')
    controller.save_adaptation_log(adaptation_log_path)

    # Save signal traces
    trace_path = os.path.join(save_dir, f'signal_traces_{timestamp}.json')
    controller.signal_collector.save_traces(trace_path)

    # Save batch results
    import json
    batch_results_path = os.path.join(save_dir, f'batch_results_{timestamp}.json')
    with open(batch_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {save_dir}")

    # Generate plots
    if not args.no_plots:
        try:
            plot_path = os.path.join(save_dir, f'robustness_traces_{timestamp}.png')
            controller.monitor.plot_robustness_traces(save_path=plot_path)
        except Exception as e:
            print(f"Could not generate plots: {e}")


def main():
    """Main execution"""
    args = parse_args()

    print("="*70)
    print("Runtime STL Monitoring Demo for TransAxx")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Results: {args.save_dir}")
    print("="*70)

    # Setup configuration
    config = setup_monitoring(args.scenario, args.save_dir)

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    val_data, calib_data = cifar10_data_loader(args.data, batch_size=args.batch_size)
    print(f"Validation batches: {len(val_data)}")
    print(f"Calibration batches: {len(calib_data)}")

    # Load model
    print(f"\nLoading model: {args.model}...")
    try:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                              args.model, pretrained=True).to(args.device)
    except Exception as e:
        print(f"Error loading model from torch hub: {e}")
        print("Please ensure you have internet connection or the model is cached")
        return

    # Initialize with approximation
    model = initialize_model_with_approximation(model, args.device)

    # Calibrate
    run_calibration(model, calib_data, args.device)

    # Create STL monitor
    print("\nInitializing STL Monitor...")
    monitor = RuntimeMonitor(config=config)

    # Create adaptive controller
    print("Initializing Adaptive Controller...")
    controller = AdaptiveController(
        model=model,
        monitor=monitor,
        config=config,
        device=args.device
    )

    # Run monitored inference
    results = run_monitored_inference(model, val_data, calib_data, controller, args)

    # Print summaries
    controller.print_adaptation_summary()
    print(f"\nFinal Running Accuracy: {controller.signal_collector.get_running_accuracy():.4f}")

    # Save everything
    save_results(controller, results, args)

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Total Adaptations: {len(results['adaptations'])}")
    print(f"  Monitoring Overhead: {results['monitoring_overhead']*1000:.2f} ms/batch")
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
