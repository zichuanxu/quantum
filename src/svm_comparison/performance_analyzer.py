"""
Performance analysis module for SVM comparison.
Measures execution times, accuracy, and generates comparison metrics.
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PerformanceAnalyzer:
    """Analyzes and compares performance between QSVM and CSVM classifiers."""

    def __init__(self):
        """Initialize PerformanceAnalyzer."""
        self.results = {}

    def measure_execution_time(self, classifier, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Measure training and prediction execution times.

        Args:
            classifier: The classifier to time
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional, for prediction timing)

        Returns:
            Dictionary with timing results
        """
        timing_results = {}

        # Measure training time
        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        timing_results['training_time'] = training_time

        # Measure prediction time if test data provided
        if X_test is not None:
            start_time = time.time()
            predictions = classifier.predict(X_test)
            prediction_time = time.time() - start_time
            timing_results['prediction_time'] = prediction_time
            timing_results['predictions'] = predictions

        return timing_results

    def calculate_accuracy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with accuracy metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }

        # Add AUC if we have probability predictions
        try:
            if len(np.unique(y_pred)) > 1:  # Check if we have both classes
                metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = 0.5  # Default AUC for single class

        return metrics

    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Confusion Matrix",
                                   class_names: List[str] = None) -> plt.Figure:
        """
        Create confusion matrix visualization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            class_names: Names for the classes

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)

        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()
        return fig

    def create_decision_boundary_plot(self, X: np.ndarray, y: np.ndarray,
                                    classifier, title: str = "Decision Boundary",
                                    resolution: int = 100) -> plt.Figure:
        """
        Create decision boundary visualization.

        Args:
            X: Feature data (2D)
            y: Labels
            classifier: Fitted classifier with get_decision_boundary method
            title: Plot title
            resolution: Grid resolution for boundary

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        try:
            # Get decision boundary
            xx, yy, Z = classifier.get_decision_boundary(X, y, resolution)

            # Plot decision boundary
            contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
            ax.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)

            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')

            # Add colorbar
            plt.colorbar(contour, ax=ax, label='Decision Function')

            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(title)

        except Exception as e:
            # Fallback: just plot the data points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f"{title} (Boundary computation failed)")
            ax.text(0.5, 0.95, f"Error: {str(e)[:50]}...", transform=ax.transAxes,
                   ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        return fig

    def compare_performance(self, qsvm_results: Dict[str, Any],
                          csvm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance comparison.

        Args:
            qsvm_results: QSVM performance results
            csvm_results: CSVM performance results

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'timing_comparison': {
                'qsvm_training_time': qsvm_results.get('training_time', 0),
                'csvm_training_time': csvm_results.get('training_time', 0),
                'training_time_ratio': (qsvm_results.get('training_time', 1) /
                                      max(csvm_results.get('training_time', 1), 1e-10)),
                'qsvm_prediction_time': qsvm_results.get('prediction_time', 0),
                'csvm_prediction_time': csvm_results.get('prediction_time', 0)
            },
            'accuracy_comparison': {
                'qsvm_accuracy': qsvm_results.get('accuracy', 0),
                'csvm_accuracy': csvm_results.get('accuracy', 0),
                'accuracy_difference': (qsvm_results.get('accuracy', 0) -
                                      csvm_results.get('accuracy', 0)),
                'qsvm_f1': qsvm_results.get('f1_score', 0),
                'csvm_f1': csvm_results.get('f1_score', 0)
            }
        }

        # Add prediction time ratio if both are available
        if (qsvm_results.get('prediction_time', 0) > 0 and
            csvm_results.get('prediction_time', 0) > 0):
            comparison['timing_comparison']['prediction_time_ratio'] = (
                qsvm_results['prediction_time'] / csvm_results['prediction_time']
            )

        # Determine winner
        if comparison['accuracy_comparison']['accuracy_difference'] > 0.01:
            comparison['accuracy_winner'] = 'QSVM'
        elif comparison['accuracy_comparison']['accuracy_difference'] < -0.01:
            comparison['accuracy_winner'] = 'CSVM'
        else:
            comparison['accuracy_winner'] = 'Tie'

        if comparison['timing_comparison']['training_time_ratio'] < 1:
            comparison['speed_winner'] = 'QSVM'
        else:
            comparison['speed_winner'] = 'CSVM'

        return comparison

    def create_performance_comparison_plot(self, comparison_results: List[Dict[str, Any]],
                                         experiment_labels: List[str]) -> plt.Figure:
        """
        Create comprehensive performance comparison visualization.

        Args:
            comparison_results: List of comparison result dictionaries
            experiment_labels: Labels for each experiment

        Returns:
            Matplotlib figure with comparison plots
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('QSVM vs CSVM Performance Comparison', fontsize=16)

        # Extract data for plotting
        qsvm_train_times = [r['timing_comparison']['qsvm_training_time'] for r in comparison_results]
        csvm_train_times = [r['timing_comparison']['csvm_training_time'] for r in comparison_results]
        qsvm_accuracies = [r['accuracy_comparison']['qsvm_accuracy'] for r in comparison_results]
        csvm_accuracies = [r['accuracy_comparison']['csvm_accuracy'] for r in comparison_results]

        x = np.arange(len(experiment_labels))
        width = 0.35

        # Training time comparison
        bars1 = ax1.bar(x - width/2, qsvm_train_times, width, label='QSVM', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, csvm_train_times, width, label='CSVM', alpha=0.8, color='red')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(experiment_labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Accuracy comparison
        bars3 = ax2.bar(x - width/2, qsvm_accuracies, width, label='QSVM', alpha=0.8, color='blue')
        bars4 = ax2.bar(x + width/2, csvm_accuracies, width, label='CSVM', alpha=0.8, color='red')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(experiment_labels, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars4:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Time ratio plot
        time_ratios = [r['timing_comparison']['training_time_ratio'] for r in comparison_results]
        bars5 = ax3.bar(x, time_ratios, alpha=0.8, color='green')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('QSVM Time / CSVM Time')
        ax3.set_title('Training Time Ratio (QSVM/CSVM)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(experiment_labels, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars5:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Accuracy difference plot
        acc_diffs = [r['accuracy_comparison']['accuracy_difference'] for r in comparison_results]
        colors = ['green' if diff > 0 else 'red' for diff in acc_diffs]
        bars6 = ax4.bar(x, acc_diffs, alpha=0.8, color=colors)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('QSVM Accuracy - CSVM Accuracy')
        ax4.set_title('Accuracy Difference (QSVM - CSVM)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(experiment_labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars6:
            height = bar.get_height()
            ax4.annotate(f'{height:+.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

        plt.tight_layout()
        return fig

    def run_complete_analysis(self, qsvm_classifier, csvm_classifier,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            experiment_name: str = "SVM Comparison") -> Dict[str, Any]:
        """
        Run complete performance analysis for both classifiers.

        Args:
            qsvm_classifier: QSVM classifier instance
            csvm_classifier: CSVM classifier instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            experiment_name: Name for this experiment

        Returns:
            Dictionary with complete analysis results
        """
        results = {'experiment_name': experiment_name}

        # Measure QSVM performance
        print(f"Analyzing QSVM performance for {experiment_name}...")
        qsvm_timing = self.measure_execution_time(qsvm_classifier, X_train, y_train, X_test)
        qsvm_predictions = qsvm_timing['predictions']
        qsvm_metrics = self.calculate_accuracy_metrics(y_test, qsvm_predictions)

        results['qsvm'] = {**qsvm_timing, **qsvm_metrics}

        # Measure CSVM performance
        print(f"Analyzing CSVM performance for {experiment_name}...")
        csvm_timing = self.measure_execution_time(csvm_classifier, X_train, y_train, X_test)
        csvm_predictions = csvm_timing['predictions']
        csvm_metrics = self.calculate_accuracy_metrics(y_test, csvm_predictions)

        results['csvm'] = {**csvm_timing, **csvm_metrics}

        # Generate comparison
        results['comparison'] = self.compare_performance(results['qsvm'], results['csvm'])

        # Store for later use
        self.results[experiment_name] = results

        return results

    def save_all_plots(self, save_directory: Path, experiment_name: str,
                      X_test: np.ndarray, y_test: np.ndarray,
                      qsvm_classifier, csvm_classifier) -> Dict[str, str]:
        """
        Generate and save all visualization plots.

        Args:
            save_directory: Directory to save plots
            experiment_name: Name for the experiment
            X_test: Test features
            y_test: Test labels
            qsvm_classifier: Fitted QSVM classifier
            csvm_classifier: Fitted CSVM classifier

        Returns:
            Dictionary mapping plot types to file paths
        """
        saved_plots = {}

        # Ensure directory exists
        save_directory.mkdir(parents=True, exist_ok=True)

        # Get predictions
        qsvm_pred = qsvm_classifier.predict(X_test)
        csvm_pred = csvm_classifier.predict(X_test)

        # Confusion matrices
        qsvm_cm_fig = self.create_confusion_matrix_plot(y_test, qsvm_pred,
                                                       f"QSVM Confusion Matrix - {experiment_name}")
        qsvm_cm_path = save_directory / f"qsvm_confusion_matrix_{experiment_name.replace(' ', '_')}.png"
        qsvm_cm_fig.savefig(qsvm_cm_path, dpi=300, bbox_inches='tight')
        plt.close(qsvm_cm_fig)
        saved_plots['qsvm_confusion_matrix'] = str(qsvm_cm_path)

        csvm_cm_fig = self.create_confusion_matrix_plot(y_test, csvm_pred,
                                                       f"CSVM Confusion Matrix - {experiment_name}")
        csvm_cm_path = save_directory / f"csvm_confusion_matrix_{experiment_name.replace(' ', '_')}.png"
        csvm_cm_fig.savefig(csvm_cm_path, dpi=300, bbox_inches='tight')
        plt.close(csvm_cm_fig)
        saved_plots['csvm_confusion_matrix'] = str(csvm_cm_path)

        # Decision boundaries
        qsvm_db_fig = self.create_decision_boundary_plot(X_test, y_test, qsvm_classifier,
                                                        f"QSVM Decision Boundary - {experiment_name}")
        qsvm_db_path = save_directory / f"qsvm_decision_boundary_{experiment_name.replace(' ', '_')}.png"
        qsvm_db_fig.savefig(qsvm_db_path, dpi=300, bbox_inches='tight')
        plt.close(qsvm_db_fig)
        saved_plots['qsvm_decision_boundary'] = str(qsvm_db_path)

        csvm_db_fig = self.create_decision_boundary_plot(X_test, y_test, csvm_classifier,
                                                        f"CSVM Decision Boundary - {experiment_name}")
        csvm_db_path = save_directory / f"csvm_decision_boundary_{experiment_name.replace(' ', '_')}.png"
        csvm_db_fig.savefig(csvm_db_path, dpi=300, bbox_inches='tight')
        plt.close(csvm_db_fig)
        saved_plots['csvm_decision_boundary'] = str(csvm_db_path)

        return saved_plots