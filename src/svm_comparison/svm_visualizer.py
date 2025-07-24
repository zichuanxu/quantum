"""
SVM visualization module for quantum and classical SVM comparison.
Creates decision boundary plots, performance charts, and comparison visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import time


class SVMVisualizer:
    """Creates visualizations for SVM comparison analysis."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize SVMVisualizer.

        Args:
            figsize: Default figure size
            dpi: Resolution for saved images
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('default')
        sns.set_palette("husl")

    def create_side_by_side_decision_boundaries(self, X: np.ndarray, y: np.ndarray,
                                              qsvm_classifier, csvm_classifier,
                                              digit_pair: Tuple[int, int],
                                              resolution: int = 50) -> plt.Figure:
        """
        Create side-by-side decision boundary comparison.

        Args:
            X: Feature data (2D)
            y: Labels
            qsvm_classifier: Fitted QSVM classifier
            csvm_classifier: Fitted CSVM classifier
            digit_pair: Tuple of (digit1, digit2)
            resolution: Grid resolution

        Returns:
            Matplotlib figure with side-by-side plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Decision Boundaries Comparison: Digits {digit_pair[0]} vs {digit_pair[1]}',
                    fontsize=16)

        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))

        # QSVM plot
        try:
            print(f"      Computing QSVM decision boundary...")
            start_time = time.time()
            xx_qsvm, yy_qsvm, Z_qsvm = qsvm_classifier.get_decision_boundary(X, y, resolution)
            elapsed = time.time() - start_time
            print(f"      QSVM boundary computed in {elapsed:.2f}s")

            contour1 = ax1.contourf(xx_qsvm, yy_qsvm, Z_qsvm, levels=20, alpha=0.6, cmap='RdYlBu')
            ax1.contour(xx_qsvm, yy_qsvm, Z_qsvm, levels=[0], colors='black', linestyles='--', linewidths=2)
        except Exception as e:
            print(f"      QSVM decision boundary failed: {e}")
            # Fallback: create a simple background
            Z_fallback = np.zeros_like(xx)
            ax1.contourf(xx, yy, Z_fallback, levels=20, alpha=0.3, cmap='RdYlBu')

        # Plot QSVM data points
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                              edgecolors='black', s=50, alpha=0.8)
        ax1.set_title('Quantum SVM (QSVM)')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.grid(True, alpha=0.3)

        # CSVM plot
        try:
            print(f"      Computing CSVM decision boundary...")
            start_time = time.time()
            xx_csvm, yy_csvm, Z_csvm = csvm_classifier.get_decision_boundary(X, y, resolution)
            elapsed = time.time() - start_time
            print(f"      CSVM boundary computed in {elapsed:.2f}s")

            contour2 = ax2.contourf(xx_csvm, yy_csvm, Z_csvm, levels=20, alpha=0.6, cmap='RdYlBu')
            ax2.contour(xx_csvm, yy_csvm, Z_csvm, levels=[0], colors='black', linestyles='--', linewidths=2)
        except Exception as e:
            print(f"      CSVM decision boundary failed: {e}")
            # Fallback: create a simple background
            Z_fallback = np.zeros_like(xx)
            ax2.contourf(xx, yy, Z_fallback, levels=20, alpha=0.3, cmap='RdYlBu')

        # Plot CSVM data points
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                              edgecolors='black', s=50, alpha=0.8)
        ax2.set_title('Classical SVM (CSVM)')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='blue', markersize=10, label=f'Digit {digit_pair[0]}'),
                          plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='red', markersize=10, label=f'Digit {digit_pair[1]}')]
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig

    def create_performance_metrics_chart(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Create a comprehensive performance metrics chart.

        Args:
            results: Results dictionary from performance analysis

        Returns:
            Matplotlib figure with performance metrics
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16)

        # Extract metrics
        qsvm_metrics = results['qsvm']
        csvm_metrics = results['csvm']

        # Timing comparison
        times = [qsvm_metrics.get('training_time', 0), csvm_metrics.get('training_time', 0)]
        classifiers = ['QSVM', 'CSVM']
        colors = ['blue', 'red']

        bars1 = ax1.bar(classifiers, times, color=colors, alpha=0.7)
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Comparison')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, time_val in zip(bars1, times):
            ax1.annotate(f'{time_val:.4f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Accuracy comparison
        accuracies = [qsvm_metrics.get('accuracy', 0), csvm_metrics.get('accuracy', 0)]
        bars2 = ax2.bar(classifiers, accuracies, color=colors, alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, acc_val in zip(bars2, accuracies):
            ax2.annotate(f'{acc_val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Multiple metrics comparison
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        qsvm_values = [qsvm_metrics.get('accuracy', 0), qsvm_metrics.get('precision', 0),
                      qsvm_metrics.get('recall', 0), qsvm_metrics.get('f1_score', 0)]
        csvm_values = [csvm_metrics.get('accuracy', 0), csvm_metrics.get('precision', 0),
                      csvm_metrics.get('recall', 0), csvm_metrics.get('f1_score', 0)]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars3 = ax3.bar(x - width/2, qsvm_values, width, label='QSVM', color='blue', alpha=0.7)
        bars4 = ax3.bar(x + width/2, csvm_values, width, label='CSVM', color='red', alpha=0.7)

        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Detailed Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # Performance ratio plot
        ratios = []
        ratio_labels = []

        if csvm_metrics.get('training_time', 0) > 0:
            time_ratio = qsvm_metrics.get('training_time', 1) / csvm_metrics.get('training_time', 1)
            ratios.append(time_ratio)
            ratio_labels.append('Time Ratio\n(QSVM/CSVM)')

        if csvm_metrics.get('accuracy', 0) > 0:
            acc_ratio = qsvm_metrics.get('accuracy', 1) / csvm_metrics.get('accuracy', 1)
            ratios.append(acc_ratio)
            ratio_labels.append('Accuracy Ratio\n(QSVM/CSVM)')

        if ratios:
            bars5 = ax4.bar(ratio_labels, ratios, color=['orange', 'green'], alpha=0.7)
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
            ax4.set_ylabel('Ratio')
            ax4.set_title('Performance Ratios')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Add value labels
            for bar, ratio_val in zip(bars5, ratios):
                ax4.annotate(f'{ratio_val:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def create_confusion_matrices_comparison(self, y_test: np.ndarray,
                                           qsvm_pred: np.ndarray, csvm_pred: np.ndarray,
                                           digit_pair: Tuple[int, int]) -> plt.Figure:
        """
        Create side-by-side confusion matrices comparison.

        Args:
            y_test: True labels
            qsvm_pred: QSVM predictions
            csvm_pred: CSVM predictions
            digit_pair: Tuple of (digit1, digit2)

        Returns:
            Matplotlib figure with confusion matrices
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Confusion Matrices: Digits {digit_pair[0]} vs {digit_pair[1]}', fontsize=16)

        # QSVM confusion matrix
        cm_qsvm = confusion_matrix(y_test, qsvm_pred)
        sns.heatmap(cm_qsvm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Digit {digit_pair[0]}', f'Digit {digit_pair[1]}'],
                   yticklabels=[f'Digit {digit_pair[0]}', f'Digit {digit_pair[1]}'], ax=ax1)
        ax1.set_title('QSVM Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # CSVM confusion matrix
        cm_csvm = confusion_matrix(y_test, csvm_pred)
        sns.heatmap(cm_csvm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=[f'Digit {digit_pair[0]}', f'Digit {digit_pair[1]}'],
                   yticklabels=[f'Digit {digit_pair[0]}', f'Digit {digit_pair[1]}'], ax=ax2)
        ax2.set_title('CSVM Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        return fig

    def create_multi_experiment_summary(self, all_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Create summary visualization for multiple experiments.

        Args:
            all_results: Dictionary mapping experiment names to results

        Returns:
            Matplotlib figure with multi-experiment summary
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Experiment Summary: QSVM vs CSVM', fontsize=16)

        # Extract data
        experiment_names = list(all_results.keys())
        qsvm_times = [all_results[exp]['qsvm']['training_time'] for exp in experiment_names]
        csvm_times = [all_results[exp]['csvm']['training_time'] for exp in experiment_names]
        qsvm_accs = [all_results[exp]['qsvm']['accuracy'] for exp in experiment_names]
        csvm_accs = [all_results[exp]['csvm']['accuracy'] for exp in experiment_names]

        x = np.arange(len(experiment_names))
        width = 0.35

        # Training times
        bars1 = ax1.bar(x - width/2, qsvm_times, width, label='QSVM', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, csvm_times, width, label='CSVM', alpha=0.8, color='red')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Across Experiments')
        ax1.set_xticks(x)
        ax1.set_xticklabels(experiment_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracies
        bars3 = ax2.bar(x - width/2, qsvm_accs, width, label='QSVM', alpha=0.8, color='blue')
        bars4 = ax2.bar(x + width/2, csvm_accs, width, label='CSVM', alpha=0.8, color='red')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Across Experiments')
        ax2.set_xticks(x)
        ax2.set_xticklabels(experiment_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Time ratios
        time_ratios = [qt/max(ct, 1e-10) for qt, ct in zip(qsvm_times, csvm_times)]
        bars5 = ax3.bar(x, time_ratios, alpha=0.8, color='orange')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('QSVM Time / CSVM Time')
        ax3.set_title('Training Time Ratios')
        ax3.set_xticks(x)
        ax3.set_xticklabels(experiment_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Accuracy differences
        acc_diffs = [qa - ca for qa, ca in zip(qsvm_accs, csvm_accs)]
        colors = ['green' if diff > 0 else 'red' for diff in acc_diffs]
        bars6 = ax4.bar(x, acc_diffs, alpha=0.8, color=colors)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('QSVM Accuracy - CSVM Accuracy')
        ax4.set_title('Accuracy Differences')
        ax4.set_xticks(x)
        ax4.set_xticklabels(experiment_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_kernel_comparison_plot(self, kernel_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Create visualization comparing different kernel performances.

        Args:
            kernel_results: Dictionary mapping kernel names to results

        Returns:
            Matplotlib figure with kernel comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Classical SVM Kernel Comparison', fontsize=16)

        kernels = list(kernel_results.keys())
        accuracies = [kernel_results[k]['accuracy'] for k in kernels]
        times = [kernel_results[k]['fit_time'] for k in kernels]

        # Accuracy comparison
        bars1 = ax1.bar(kernels, accuracies, alpha=0.8, color=['blue', 'red', 'green', 'orange'])
        ax1.set_xlabel('Kernel Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Kernel Type')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Training time comparison
        bars2 = ax2.bar(kernels, times, alpha=0.8, color=['blue', 'red', 'green', 'orange'])
        ax2.set_xlabel('Kernel Type')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time by Kernel Type')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, time_val in zip(bars2, times):
            ax2.annotate(f'{time_val:.4f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def save_visualization(self, fig: plt.Figure, filename: str, directory: Path) -> str:
        """
        Save a matplotlib figure to the specified directory.

        Args:
            fig: Matplotlib figure to save
            filename: Name of the file (without extension)
            directory: Directory to save the file

        Returns:
            Full path to the saved file
        """
        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # Add .png extension if not present
        if not filename.endswith('.png'):
            filename += '.png'

        filepath = directory / filename

        # Save the figure
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        # Close the figure to free memory
        plt.close(fig)

        return str(filepath)