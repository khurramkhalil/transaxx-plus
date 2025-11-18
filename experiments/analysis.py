"""
Analysis and Visualization Tools for Experimental Results

Creates publication-ready plots and tables from experiment data.
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class ResultAnalyzer:
    """
    Analyzer for experimental results.

    Generates:
    - Comparison bar charts
    - Robustness traces
    - Accuracy vs. power Pareto curves
    - Adaptation timelines
    - Statistical summaries
    """

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.results = {}
        self._load_results()

        # Set publication style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

    def _load_results(self):
        """Load all experiment results from directory"""
        for filename in os.listdir(self.results_dir):
            if filename.endswith('_result.json'):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    name = data['config']['name']
                    self.results[name] = data

        print(f"Loaded {len(self.results)} experiment results")

    def plot_accuracy_comparison(self, save_path: Optional[str] = None):
        """Bar chart comparing final accuracy across approaches"""
        approaches = []
        accuracies = []

        for name, result in self.results.items():
            approaches.append(name)
            accuracies.append(result['metrics'].get('final_accuracy', 0))

        plt.figure(figsize=(10, 6))
        bars = plt.bar(approaches, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')

        plt.xlabel('Approach')
        plt.ylabel('Final Accuracy')
        plt.title('Accuracy Comparison Across Approaches')
        plt.ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])
        plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy comparison to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_adaptations_comparison(self, save_path: Optional[str] = None):
        """Bar chart comparing number of adaptations"""
        approaches = []
        adaptations = []

        for name, result in self.results.items():
            approaches.append(name)
            adaptations.append(len(result.get('adaptations', [])))

        plt.figure(figsize=(10, 6))
        bars = plt.bar(approaches, adaptations, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        plt.xlabel('Approach')
        plt.ylabel('Number of Adaptations')
        plt.title('Adaptation Frequency Comparison')
        plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved adaptations comparison to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_violations_comparison(self, save_path: Optional[str] = None):
        """Bar chart comparing violations and warnings"""
        approaches = []
        violations = []
        warnings = []

        for name, result in self.results.items():
            approaches.append(name)
            violations.append(len(result.get('violations', [])))
            warnings.append(len(result.get('warnings', [])))

        x = np.arange(len(approaches))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, violations, width, label='Violations', color='#e74c3c')
        plt.bar(x + width/2, warnings, width, label='Warnings', color='#f39c12')

        plt.xlabel('Approach')
        plt.ylabel('Count')
        plt.title('Violations and Warnings Comparison')
        plt.xticks(x, approaches)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved violations comparison to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_accuracy_over_time(self, save_path: Optional[str] = None):
        """Line plot showing accuracy evolution over batches"""
        plt.figure(figsize=(12, 6))

        for name, result in self.results.items():
            batch_results = result.get('batch_results', [])
            if batch_results:
                batches = [br['batch_idx'] for br in batch_results]
                accuracies = [br['accuracy'] for br in batch_results]
                plt.plot(batches, accuracies, label=name, linewidth=2, marker='o', markersize=3)

        plt.xlabel('Batch Index')
        plt.ylabel('Batch Accuracy')
        plt.title('Accuracy Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy over time to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_accuracy_vs_power(self, save_path: Optional[str] = None):
        """Scatter plot: accuracy vs. power consumption (Pareto frontier)"""
        plt.figure(figsize=(10, 8))

        for name, result in self.results.items():
            batch_results = result.get('batch_results', [])
            if batch_results:
                accuracies = [br['accuracy'] for br in batch_results]
                powers = [br.get('power', 0.7) for br in batch_results]

                avg_accuracy = np.mean(accuracies)
                avg_power = np.mean(powers)

                plt.scatter(avg_power, avg_accuracy, s=200, label=name, alpha=0.7)
                plt.annotate(name, (avg_power, avg_accuracy),
                           xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Average Power Consumption (relative)')
        plt.ylabel('Average Accuracy')
        plt.title('Accuracy vs. Power Trade-off (Pareto Analysis)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy vs. power to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_robustness_traces(self, stl_experiment_name: str, save_path: Optional[str] = None):
        """Plot robustness traces for STL experiment"""
        if stl_experiment_name not in self.results:
            print(f"Experiment '{stl_experiment_name}' not found")
            return

        result = self.results[stl_experiment_name]
        batch_results = result.get('batch_results', [])

        if not batch_results or 'robustness' not in batch_results[0]:
            print(f"No robustness data in experiment '{stl_experiment_name}'")
            return

        # Extract robustness traces for each formula
        formulas = batch_results[0]['robustness'].keys()
        traces = {formula: [] for formula in formulas}

        for br in batch_results:
            for formula in formulas:
                traces[formula].append(br['robustness'].get(formula, 0))

        # Plot
        n_formulas = len(formulas)
        fig, axes = plt.subplots(n_formulas, 1, figsize=(12, 3*n_formulas), sharex=True)

        if n_formulas == 1:
            axes = [axes]

        for ax, (formula, trace) in zip(axes, traces.items()):
            batches = list(range(len(trace)))
            ax.plot(batches, trace, linewidth=2, color='#3498db')
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Violation threshold')
            ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, label='Warning threshold')
            ax.fill_between(batches, 0, trace,
                           where=[t >= 0 for t in trace],
                           alpha=0.3, color='green', label='Satisfied')
            ax.fill_between(batches, trace, 0,
                           where=[t < 0 for t in trace],
                           alpha=0.3, color='red', label='Violated')
            ax.set_ylabel('Robustness ρ')
            ax.set_title(f'Formula: {formula}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Batch Index')
        plt.suptitle(f'STL Robustness Traces - {stl_experiment_name}', y=1.001)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved robustness traces to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_latex_table(self, save_path: Optional[str] = None):
        """Generate LaTeX table for publication"""
        table_lines = []
        table_lines.append(r"\begin{table}[htbp]")
        table_lines.append(r"\centering")
        table_lines.append(r"\caption{Comparison of Adaptation Approaches}")
        table_lines.append(r"\label{tab:comparison}")
        table_lines.append(r"\begin{tabular}{lccccc}")
        table_lines.append(r"\toprule")
        table_lines.append(r"Approach & Accuracy & Adaptations & Violations & Warnings & Time (s) \\")
        table_lines.append(r"\midrule")

        for name, result in self.results.items():
            metrics = result['metrics']
            accuracy = metrics.get('final_accuracy', 0)
            adaptations = len(result.get('adaptations', []))
            violations = len(result.get('violations', []))
            warnings = len(result.get('warnings', []))
            time_taken = result['timing'].get('total_time', 0)

            table_lines.append(
                f"{name} & {accuracy:.3f} & {adaptations} & {violations} & {warnings} & {time_taken:.1f} \\\\"
            )

        table_lines.append(r"\bottomrule")
        table_lines.append(r"\end{tabular}")
        table_lines.append(r"\end{table}")

        latex_table = "\n".join(table_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex_table)
            print(f"Saved LaTeX table to: {save_path}")
        else:
            print(latex_table)

        return latex_table

    def generate_summary_report(self, save_path: Optional[str] = None):
        """Generate comprehensive text summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENTAL RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        for name, result in self.results.items():
            lines.append(f"\n{name.upper()}")
            lines.append("-" * 40)
            metrics = result['metrics']
            lines.append(f"  Final Accuracy:      {metrics.get('final_accuracy', 0):.4f}")
            lines.append(f"  Total Batches:       {metrics.get('total_batches', 0)}")
            lines.append(f"  Adaptations:         {len(result.get('adaptations', []))}")
            lines.append(f"  Violations:          {len(result.get('violations', []))}")
            lines.append(f"  Warnings:            {len(result.get('warnings', []))}")
            lines.append(f"  Total Time:          {result['timing'].get('total_time', 0):.2f}s")
            lines.append(f"  Avg Time/Batch:      {result['timing'].get('avg_time_per_batch', 0)*1000:.2f}ms")

        lines.append("\n" + "=" * 80)
        lines.append("KEY FINDINGS:")
        lines.append("=" * 80)

        # Compute relative comparisons
        if 'static' in self.results and 'stl_balanced' in self.results:
            static_acc = self.results['static']['metrics'].get('final_accuracy', 0)
            stl_acc = self.results['stl_balanced']['metrics'].get('final_accuracy', 0)
            acc_diff = (stl_acc - static_acc) * 100

            lines.append(f"\n  STL vs. Static:")
            lines.append(f"    Accuracy difference: {acc_diff:+.2f}%")
            lines.append(f"    Adaptations: {len(self.results['stl_balanced'].get('adaptations', []))}")

        if 'threshold' in self.results and 'stl_balanced' in self.results:
            threshold_acc = self.results['threshold']['metrics'].get('final_accuracy', 0)
            stl_acc = self.results['stl_balanced']['metrics'].get('final_accuracy', 0)
            acc_diff = (stl_acc - threshold_acc) * 100

            stl_violations = len(self.results['stl_balanced'].get('violations', []))
            thresh_violations = len(self.results['threshold'].get('violations', []))

            lines.append(f"\n  STL vs. Threshold:")
            lines.append(f"    Accuracy difference: {acc_diff:+.2f}%")
            lines.append(f"    Violations (STL): {stl_violations}")
            lines.append(f"    Violations (Threshold): {thresh_violations}")
            lines.append(f"    Predictive advantage: {thresh_violations - stl_violations} fewer violations")

        lines.append("\n" + "=" * 80)

        report = "\n".join(lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved summary report to: {save_path}")
        else:
            print(report)

        return report

    def generate_all_plots(self, output_dir: Optional[str] = None):
        """Generate all plots at once"""
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'plots')

        os.makedirs(output_dir, exist_ok=True)

        print("\nGenerating all plots...")

        # Generate plots
        self.plot_accuracy_comparison(os.path.join(output_dir, 'accuracy_comparison.png'))
        self.plot_adaptations_comparison(os.path.join(output_dir, 'adaptations_comparison.png'))
        self.plot_violations_comparison(os.path.join(output_dir, 'violations_comparison.png'))
        self.plot_accuracy_over_time(os.path.join(output_dir, 'accuracy_over_time.png'))
        self.plot_accuracy_vs_power(os.path.join(output_dir, 'accuracy_vs_power.png'))

        # Generate robustness traces for STL experiments
        for name in self.results.keys():
            if name.startswith('stl_'):
                self.plot_robustness_traces(name, os.path.join(output_dir, f'robustness_{name}.png'))

        # Generate tables and reports
        self.generate_latex_table(os.path.join(output_dir, 'comparison_table.tex'))
        self.generate_summary_report(os.path.join(output_dir, 'summary_report.txt'))

        print(f"\nAll plots and reports saved to: {output_dir}")
