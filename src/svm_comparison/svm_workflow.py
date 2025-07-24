"""
Main execution workflow for SVM comparison implementation.
Orchestrates data processing, QSVM/CSVM training, performance analysis, and reporting.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import os
import numpy as np

# Add the quantum package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.svm_comparison.data_processor import DataProcessor
from src.svm_comparison.qsvm_classifier import QSVMClassifier
from src.svm_comparison.csvm_classifier import CSVMClassifier
from src.svm_comparison.performance_analyzer import PerformanceAnalyzer
from src.svm_comparison.svm_visualizer import SVMVisualizer
from src.utils.file_manager import FileManager
from src.utils.markdown_generator import SVMReportGenerator


class SVMWorkflow:
    """Main workflow for executing SVM comparison analysis."""

    def __init__(self, base_dir: str = "quantum", random_state: int = 42):
        """
        Initialize SVM workflow.

        Args:
            base_dir: Base directory for the project
            random_state: Random seed for reproducibility
        """
        self.base_dir = base_dir
        self.random_state = random_state

        # Initialize components
        self.file_manager = FileManager(base_dir)
        self.data_processor = DataProcessor(random_state)
        self.performance_analyzer = PerformanceAnalyzer()
        self.svm_visualizer = SVMVisualizer()
        self.report_generator = SVMReportGenerator()

        # Ensure directories exist
        self.file_manager.ensure_directories_exist()

        # Storage for results
        self.all_results = {}
        self.processed_datasets = {}

    def prepare_datasets(self, digit_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Prepare datasets for all digit pairs.

        Args:
            digit_pairs: List of digit pairs to process

        Returns:
            Dictionary mapping digit pairs to processed data
        """
        print("Preparing datasets...")

        # Load and process datasets
        self.processed_datasets = self.data_processor.process_complete_dataset(digit_pairs)

        print(f"✓ Prepared {len(self.processed_datasets)} datasets")
        return self.processed_datasets

    def execute_single_experiment(self, digit_pair: Tuple[int, int],
                                kernel_type: str = 'linear') -> Dict[str, Any]:
        """
        Execute a single SVM comparison experiment.

        Args:
            digit_pair: Tuple of digit classes to compare
            kernel_type: Kernel type for classical SVM

        Returns:
            Dictionary with experiment results
        """
        print(f"Executing experiment: Digits {digit_pair[0]} vs {digit_pair[1]} ({kernel_type} kernel)")

        if digit_pair not in self.processed_datasets:
            raise ValueError(f"Dataset for {digit_pair} not prepared")

        data = self.processed_datasets[digit_pair]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        X_full, y_full = data['X_full'], data['y_full']

        experiment_results = {
            'digit_pair': digit_pair,
            'kernel_type': kernel_type,
            'dataset_info': data['info']
        }

        # Initialize and train QSVM
        print("  Training QSVM...")
        try:
            qsvm = QSVMClassifier()
            qsvm_timing = self.performance_analyzer.measure_execution_time(qsvm, X_train, y_train, X_test)
            qsvm_predictions = qsvm_timing['predictions']
            qsvm_metrics = self.performance_analyzer.calculate_accuracy_metrics(y_test, qsvm_predictions)

            qsvm_results = {**qsvm_timing, **qsvm_metrics}
            experiment_results['qsvm_results'] = qsvm_results
            experiment_results['qsvm_classifier'] = qsvm
            print(f"    ✓ QSVM completed (Accuracy: {qsvm_results['accuracy']:.3f})")
        except Exception as e:
            print(f"    ✗ QSVM training failed: {e}")
            experiment_results['qsvm_results'] = {
                'error': str(e),
                'training_time': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            experiment_results['qsvm_classifier'] = None

        # Initialize and train CSVM
        print("  Training CSVM...")
        try:
            csvm = CSVMClassifier(kernel=kernel_type, random_state=self.random_state)
            csvm_timing = self.performance_analyzer.measure_execution_time(csvm, X_train, y_train, X_test)
            csvm_predictions = csvm_timing['predictions']
            csvm_metrics = self.performance_analyzer.calculate_accuracy_metrics(y_test, csvm_predictions)

            csvm_results = {**csvm_timing, **csvm_metrics}
            experiment_results['csvm_results'] = csvm_results
            experiment_results['csvm_classifier'] = csvm
            print(f"    ✓ CSVM completed (Accuracy: {csvm_results['accuracy']:.3f})")
        except Exception as e:
            print(f"    ✗ CSVM training failed: {e}")
            experiment_results['csvm_results'] = {
                'error': str(e),
                'training_time': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            experiment_results['csvm_classifier'] = None

        # Generate performance comparison
        if ('qsvm_results' in experiment_results and 'csvm_results' in experiment_results and
            'error' not in experiment_results['qsvm_results'] and
            'error' not in experiment_results['csvm_results']):
            comparison = self.performance_analyzer.compare_performance(
                experiment_results['qsvm_results'],
                experiment_results['csvm_results']
            )
            experiment_results['comparison'] = comparison

        # Create visualizations (with timeout protection)
        try:
            print("  Creating visualizations...")

            # Performance comparison plot (quick and reliable)
            if ('qsvm_results' in experiment_results and 'csvm_results' in experiment_results):
                try:
                    print("    Creating performance chart...")
                    performance_fig = self.svm_visualizer.create_performance_metrics_chart({
                        'qsvm': experiment_results['qsvm_results'],
                        'csvm': experiment_results['csvm_results']
                    })
                    performance_path = self.svm_visualizer.save_visualization(
                        performance_fig, f"performance_{digit_pair[0]}vs{digit_pair[1]}_{kernel_type}",
                        self.file_manager.svm_images_dir
                    )
                    experiment_results['performance_viz_path'] = performance_path
                    print("    ✓ Performance chart created")
                except Exception as e:
                    print(f"    Warning: Performance chart failed: {e}")

            # Decision boundary comparison (potentially slow, with reduced resolution)
            if (experiment_results.get('qsvm_classifier') and
                experiment_results.get('csvm_classifier')):
                try:
                    print("    Creating decision boundaries (reduced resolution)...")
                    # Use smaller dataset and lower resolution for faster visualization
                    sample_size = min(200, len(X_full))
                    indices = np.random.choice(len(X_full), sample_size, replace=False)
                    X_sample = X_full[indices]
                    y_sample = y_full[indices]

                    boundary_fig = self.svm_visualizer.create_side_by_side_decision_boundaries(
                        X_sample, y_sample,
                        experiment_results['qsvm_classifier'],
                        experiment_results['csvm_classifier'],
                        digit_pair,
                        resolution=50  # Reduced resolution for speed
                    )
                    boundary_path = self.svm_visualizer.save_visualization(
                        boundary_fig, f"boundaries_{digit_pair[0]}vs{digit_pair[1]}_{kernel_type}",
                        self.file_manager.svm_images_dir
                    )
                    experiment_results['boundary_viz_path'] = boundary_path
                    print("    ✓ Decision boundaries created")
                except Exception as e:
                    print(f"    Warning: Decision boundary visualization failed: {e}")

            # Confusion matrices comparison (quick)
            if (experiment_results.get('qsvm_classifier') and
                experiment_results.get('csvm_classifier')):
                try:
                    print("    Creating confusion matrices...")
                    qsvm_pred = experiment_results['qsvm_classifier'].predict(X_test)
                    csvm_pred = experiment_results['csvm_classifier'].predict(X_test)

                    confusion_fig = self.svm_visualizer.create_confusion_matrices_comparison(
                        y_test, qsvm_pred, csvm_pred, digit_pair
                    )
                    confusion_path = self.svm_visualizer.save_visualization(
                        confusion_fig, f"confusion_{digit_pair[0]}vs{digit_pair[1]}_{kernel_type}",
                        self.file_manager.svm_images_dir
                    )
                    experiment_results['confusion_viz_path'] = confusion_path
                    print("    ✓ Confusion matrices created")
                except Exception as e:
                    print(f"    Warning: Confusion matrix visualization failed: {e}")

        except Exception as e:
            print(f"  Warning: Visualization creation failed: {e}")

        print(f"  ✓ Experiment completed")
        return experiment_results

    def execute_kernel_comparison(self, digit_pair: Tuple[int, int],
                                kernels: List[str] = None) -> Dict[str, Any]:
        """
        Execute kernel comparison for a specific digit pair.

        Args:
            digit_pair: Tuple of digit classes
            kernels: List of kernels to compare

        Returns:
            Dictionary with kernel comparison results
        """
        if kernels is None:
            kernels = ['linear', 'rbf', 'poly']

        print(f"Executing kernel comparison for digits {digit_pair[0]} vs {digit_pair[1]}")

        if digit_pair not in self.processed_datasets:
            raise ValueError(f"Dataset for {digit_pair} not prepared")

        data = self.processed_datasets[digit_pair]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        kernel_results = {}

        for kernel in kernels:
            try:
                print(f"  Testing {kernel} kernel...")
                csvm = CSVMClassifier(kernel=kernel, random_state=self.random_state)

                # Measure performance
                timing_results = self.performance_analyzer.measure_execution_time(
                    csvm, X_train, y_train, X_test
                )
                predictions = timing_results['predictions']
                accuracy_metrics = self.performance_analyzer.calculate_accuracy_metrics(
                    y_test, predictions
                )

                kernel_results[kernel] = {
                    'accuracy': accuracy_metrics['accuracy'],
                    'fit_time': timing_results['training_time'],
                    'predict_time': timing_results.get('prediction_time', 0),
                    **accuracy_metrics
                }

            except Exception as e:
                print(f"    ✗ {kernel} kernel failed: {e}")
                kernel_results[kernel] = {
                    'error': str(e),
                    'accuracy': 0.0,
                    'fit_time': float('inf')
                }

        # Create kernel comparison visualization
        try:
            kernel_fig = self.svm_visualizer.create_kernel_comparison_plot(kernel_results)
            kernel_path = self.svm_visualizer.save_visualization(
                kernel_fig, f"kernels_{digit_pair[0]}vs{digit_pair[1]}",
                self.file_manager.svm_images_dir
            )

            return {
                'digit_pair': digit_pair,
                'kernel_results': kernel_results,
                'kernel_viz_path': kernel_path
            }

        except Exception as e:
            print(f"Warning: Kernel comparison visualization failed: {e}")
            return {
                'digit_pair': digit_pair,
                'kernel_results': kernel_results
            }

    def execute_all_experiments(self, digit_pairs: List[Tuple[int, int]],
                              kernels: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Execute all SVM comparison experiments.

        Args:
            digit_pairs: List of digit pairs to analyze
            kernels: List of kernels to test

        Returns:
            Dictionary of experiment results
        """
        if kernels is None:
            kernels = ['linear', 'rbf']

        print("Starting SVM comparison experiments...")

        # Prepare datasets
        self.prepare_datasets(digit_pairs)

        # Execute experiments
        for digit_pair in digit_pairs:
            for kernel in kernels:
                experiment_key = f"{digit_pair[0]}vs{digit_pair[1]}_{kernel}"
                try:
                    result = self.execute_single_experiment(digit_pair, kernel)
                    self.all_results[experiment_key] = result
                except Exception as e:
                    print(f"✗ Experiment failed for {digit_pair} with {kernel}: {e}")
                    self.all_results[experiment_key] = {
                        'digit_pair': digit_pair,
                        'kernel_type': kernel,
                        'error': str(e)
                    }

        # Execute kernel comparisons
        for digit_pair in digit_pairs:
            try:
                kernel_comparison = self.execute_kernel_comparison(digit_pair, kernels + ['poly'])
                self.all_results[f"kernel_comparison_{digit_pair[0]}vs{digit_pair[1]}"] = kernel_comparison
            except Exception as e:
                print(f"Warning: Kernel comparison failed for {digit_pair}: {e}")

        return self.all_results

    def create_summary_visualizations(self) -> Dict[str, str]:
        """
        Create summary visualizations across all experiments.

        Returns:
            Dictionary mapping visualization types to file paths
        """
        print("Creating summary visualizations...")

        summary_paths = {}

        try:
            # Filter successful experiment results (not kernel comparisons)
            experiment_results = {k: v for k, v in self.all_results.items()
                                if not k.startswith('kernel_comparison') and 'error' not in v}

            if experiment_results:
                # Multi-experiment summary
                summary_fig = self.svm_visualizer.create_multi_experiment_summary(experiment_results)
                summary_path = self.svm_visualizer.save_visualization(
                    summary_fig, "multi_experiment_summary", self.file_manager.svm_images_dir
                )
                summary_paths['multi_experiment_summary'] = summary_path

        except Exception as e:
            print(f"Warning: Summary visualization creation failed: {e}")

        return summary_paths

    def generate_markdown_report(self) -> str:
        """
        Generate comprehensive markdown report for SVM comparison.

        Returns:
            Path to the generated markdown report
        """
        print("Generating SVM comparison report...")

        # Initialize report
        self.report_generator.create_svm_report_template()

        # Add individual experiment results
        experiment_results = {k: v for k, v in self.all_results.items()
                            if not k.startswith('kernel_comparison') and 'error' not in v}

        for exp_key, result in experiment_results.items():
            digit_pair = result['digit_pair']
            kernel_type = result['kernel_type']
            qsvm_results = result.get('qsvm_results', {})
            csvm_results = result.get('csvm_results', {})

            # Get relative path for decision boundary image
            boundary_image = ""
            if 'boundary_viz_path' in result:
                boundary_image = self.file_manager.get_relative_image_path(
                    "svm", Path(result['boundary_viz_path']).name
                )

            self.report_generator.add_experiment_results(
                digit_pair, kernel_type, qsvm_results, csvm_results, boundary_image
            )

        # Add kernel comparison results
        kernel_comparisons = {k: v for k, v in self.all_results.items()
                            if k.startswith('kernel_comparison')}

        if kernel_comparisons:
            self.report_generator.add_header("Kernel Comparison Analysis", 2)
            for comp_key, comp_result in kernel_comparisons.items():
                digit_pair = comp_result['digit_pair']
                kernel_results = comp_result['kernel_results']

                self.report_generator.add_header(f"Kernels for Digits {digit_pair[0]} vs {digit_pair[1]}", 3)

                # Create comparison table
                headers = ["Kernel", "Accuracy", "Training Time (s)", "F1-Score"]
                rows = []
                for kernel, metrics in kernel_results.items():
                    if 'error' not in metrics:
                        rows.append([
                            kernel.capitalize(),
                            f"{metrics['accuracy']:.3f}",
                            f"{metrics['fit_time']:.4f}",
                            f"{metrics.get('f1_score', 0):.3f}"
                        ])

                if rows:
                    self.report_generator.add_table(headers, rows)

                # Add kernel comparison image if available
                if 'kernel_viz_path' in comp_result:
                    kernel_image = self.file_manager.get_relative_image_path(
                        "svm", Path(comp_result['kernel_viz_path']).name
                    )
                    self.report_generator.add_image(
                        f"Kernel Comparison for Digits {digit_pair[0]} vs {digit_pair[1]}",
                        kernel_image,
                        f"Performance comparison across different kernel types"
                    )

        # Add summary analysis
        if experiment_results:
            # Calculate summary statistics
            qsvm_accuracies = [r['qsvm_results']['accuracy'] for r in experiment_results.values()
                             if 'qsvm_results' in r and 'error' not in r['qsvm_results']]
            csvm_accuracies = [r['csvm_results']['accuracy'] for r in experiment_results.values()
                             if 'csvm_results' in r and 'error' not in r['csvm_results']]
            qsvm_times = [r['qsvm_results']['training_time'] for r in experiment_results.values()
                         if 'qsvm_results' in r and 'error' not in r['qsvm_results']]
            csvm_times = [r['csvm_results']['training_time'] for r in experiment_results.values()
                         if 'csvm_results' in r and 'error' not in r['csvm_results']]

            summary_data = []
            for exp_key, result in experiment_results.items():
                if ('qsvm_results' in result and 'csvm_results' in result and
                    'error' not in result['qsvm_results'] and 'error' not in result['csvm_results']):
                    summary_data.append({
                        'digit_pair': result['digit_pair'],
                        'qsvm_time': result['qsvm_results']['training_time'],
                        'csvm_time': result['csvm_results']['training_time'],
                        'qsvm_accuracy': result['qsvm_results']['accuracy'],
                        'csvm_accuracy': result['csvm_results']['accuracy']
                    })

            if summary_data:
                self.report_generator.add_summary_analysis(summary_data)

        # Add summary visualizations
        summary_paths = self.create_summary_visualizations()
        if summary_paths:
            self.report_generator.add_header("Summary Visualizations", 2)

            for viz_type, path in summary_paths.items():
                relative_path = self.file_manager.get_relative_image_path("svm", Path(path).name)
                self.report_generator.add_image(
                    f"{viz_type.replace('_', ' ').title()}",
                    relative_path,
                    f"Comprehensive analysis across all experiments"
                )

        # Add conclusions
        self.report_generator.add_header("Key Findings and Conclusions", 2)

        if experiment_results:
            # Calculate win rates
            qsvm_wins = sum(1 for r in experiment_results.values()
                           if ('comparison' in r and
                               r['comparison'].get('accuracy_winner') == 'QSVM'))
            total_experiments = len([r for r in experiment_results.values()
                                   if 'comparison' in r])

            conclusions = [
                f"Total experiments conducted: {len(experiment_results)}",
                f"QSVM accuracy wins: {qsvm_wins}/{total_experiments} experiments" if total_experiments > 0 else "No valid comparisons available",
                f"Average QSVM accuracy: {np.mean(qsvm_accuracies):.3f}" if qsvm_accuracies else "No QSVM results",
                f"Average CSVM accuracy: {np.mean(csvm_accuracies):.3f}" if csvm_accuracies else "No CSVM results",
                f"Average QSVM training time: {np.mean(qsvm_times):.4f}s" if qsvm_times else "No QSVM timing data",
                f"Average CSVM training time: {np.mean(csvm_times):.4f}s" if csvm_times else "No CSVM timing data"
            ]

            self.report_generator.add_list(conclusions)

        # Save report
        report_path = self.file_manager.get_results_path("svm_comparison.md")
        self.report_generator.save_to_file(report_path)

        print(f"✓ Report saved to: {report_path}")
        return str(report_path)

    def run_complete_analysis(self, digit_pairs: List[Tuple[int, int]] = None,
                            kernels: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete SVM comparison analysis workflow.

        Args:
            digit_pairs: List of digit pairs to analyze (default: [(3,4), (1,2)])
            kernels: List of kernels to test (default: ['linear', 'rbf'])

        Returns:
            Dictionary with summary of all results and paths
        """
        if digit_pairs is None:
            digit_pairs = [(3, 4), (1, 2)]
        if kernels is None:
            kernels = ['linear', 'rbf']

        print("=" * 60)
        print("SVM COMPARISON ANALYSIS - COMPLETE WORKFLOW")
        print("=" * 60)

        try:
            # Execute all experiments
            results = self.execute_all_experiments(digit_pairs, kernels)

            # Generate report
            report_path = self.generate_markdown_report()

            # Summary
            experiment_results = {k: v for k, v in results.items()
                                if not k.startswith('kernel_comparison')}
            successful_results = {k: v for k, v in experiment_results.items() if 'error' not in v}
            failed_results = {k: v for k, v in experiment_results.items() if 'error' in v}

            summary = {
                'total_experiments': len(experiment_results),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(failed_results),
                'success_rate': len(successful_results) / len(experiment_results) * 100 if experiment_results else 0,
                'digit_pairs_tested': digit_pairs,
                'kernels_tested': kernels,
                'report_path': report_path,
                'results': results
            }

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Total experiments: {summary['total_experiments']}")
            print(f"Successful: {summary['successful_experiments']}")
            print(f"Failed: {summary['failed_experiments']}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            print(f"Report generated: {report_path}")
            print(f"Images saved to: {self.file_manager.svm_images_dir}")

            return summary

        except Exception as e:
            print(f"✗ Complete analysis failed: {e}")
            return {
                'error': str(e),
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0,
                'success_rate': 0.0
            }


def main():
    """Main function to run SVM comparison analysis."""
    workflow = SVMWorkflow()
    summary = workflow.run_complete_analysis()
    return summary


if __name__ == "__main__":
    main()