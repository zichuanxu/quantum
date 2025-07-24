"""
Classical Support Vector Machine (CSVM) classifier implementation.
Uses scikit-learn's optimized SVM implementation with various kernels.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from typing import Optional, Dict, Any, Tuple, List
import time


class CSVMClassifier(BaseEstimator, ClassifierMixin):
    """Classical Support Vector Machine classifier with multiple kernel options."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0, gamma: str = 'scale',
                 degree: int = 3, coef0: float = 0.0, probability: bool = True,
                 random_state: int = 42):
        """
        Initialize CSVM classifier.

        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of polynomial kernel
            coef0: Independent term in kernel function
            probability: Whether to enable probability estimates
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state

        self.svm = None
        self.is_fitted = False
        self.n_features = None

    def _create_svm(self) -> SVC:
        """
        Create SVM instance with specified parameters.

        Returns:
            Configured SVC instance
        """
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probability,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSVMClassifier':
        """
        Fit the CSVM classifier to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self (fitted classifier)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features = X.shape[1]

        # Create and fit SVM
        self.svm = self._create_svm()
        self.svm.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X: Test features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        return self.svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test features

        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if self.probability:
            return self.svm.predict_proba(X)
        else:
            # Convert decision function to probabilities using sigmoid
            decision = self.svm.decision_function(X)
            prob_pos = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - prob_pos, prob_pos])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.

        Args:
            X: Test features

        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before computing decision function")

        return self.svm.decision_function(X)

    def get_decision_boundary(self, X: np.ndarray, y: np.ndarray,
                            resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute decision boundary for visualization.

        Args:
            X: Feature data for determining range
            y: Labels (not used but kept for interface consistency)
            resolution: Resolution of the boundary grid

        Returns:
            Tuple of (xx, yy, Z) where Z contains decision values
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before computing decision boundary")

        # Create mesh grid with reduced resolution for speed
        resolution = min(resolution, 50)  # Cap resolution to prevent slowdown
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))

        # Compute decision function on grid with batch processing
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        try:
            # Process in smaller batches for consistency with QSVM
            batch_size = min(1000, len(grid_points))
            Z_parts = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                Z_batch = self.decision_function(batch)
                Z_parts.append(Z_batch)

            Z = np.concatenate(Z_parts)
            Z = Z.reshape(xx.shape)

        except Exception as e:
            print(f"Warning: CSVM decision boundary computation failed ({e})")
            # Return zero decision boundary as fallback
            Z = np.zeros_like(xx)

        return xx, yy, Z

    def get_support_vectors(self) -> np.ndarray:
        """
        Get support vectors from the fitted model.

        Returns:
            Support vectors array
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before accessing support vectors")

        return self.svm.support_vectors_

    def get_support_vector_indices(self) -> np.ndarray:
        """
        Get indices of support vectors.

        Returns:
            Support vector indices
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before accessing support vector indices")

        return self.svm.support_

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.

        Returns:
            Dictionary with model information
        """
        info = {
            'classifier_type': 'CSVM',
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted
        }

        if self.is_fitted:
            info.update({
                'n_support_vectors': len(self.svm.support_),
                'n_support_vectors_per_class': self.svm.n_support_,
                'dual_coef_shape': self.svm.dual_coef_.shape if hasattr(self.svm, 'dual_coef_') else None,
                'intercept': self.svm.intercept_[0] if len(self.svm.intercept_) == 1 else self.svm.intercept_
            })

        return info

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                cv: int = 5, scoring: str = 'accuracy',
                                param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search cross-validation.

        Args:
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            param_grid: Parameter grid for search (if None, uses default)

        Returns:
            Dictionary with optimization results
        """
        if param_grid is None:
            if self.kernel == 'linear':
                param_grid = {
                    'C': [0.1, 1, 10, 100]
                }
            elif self.kernel == 'rbf':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            elif self.kernel == 'poly':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto']
                }
            else:  # sigmoid
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'coef0': [0, 1, -1]
                }

        # Create base SVM
        base_svm = self._create_svm()

        # Perform grid search
        grid_search = GridSearchCV(
            base_svm, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )

        grid_search.fit(X, y)

        # Update parameters with best found
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            setattr(self, param, value)

        # Refit with best parameters
        self.fit(X, y)

        return {
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }


class MultiKernelCSVM:
    """Wrapper class for testing multiple kernel types."""

    def __init__(self, kernels: List[str] = None, **svm_params):
        """
        Initialize multi-kernel CSVM.

        Args:
            kernels: List of kernel types to test
            **svm_params: Additional SVM parameters
        """
        if kernels is None:
            kernels = ['linear', 'rbf', 'poly', 'sigmoid']

        self.kernels = kernels
        self.svm_params = svm_params
        self.classifiers = {}
        self.results = {}

    def fit_all_kernels(self, X: np.ndarray, y: np.ndarray) -> Dict[str, CSVMClassifier]:
        """
        Fit SVM with all specified kernels.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Dictionary mapping kernel names to fitted classifiers
        """
        for kernel in self.kernels:
            print(f"Fitting CSVM with {kernel} kernel...")

            classifier = CSVMClassifier(kernel=kernel, **self.svm_params)

            start_time = time.time()
            classifier.fit(X, y)
            fit_time = time.time() - start_time

            self.classifiers[kernel] = classifier
            self.results[kernel] = {
                'classifier': classifier,
                'fit_time': fit_time,
                'model_info': classifier.get_model_info()
            }

        return self.classifiers

    def evaluate_all_kernels(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all fitted kernels on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation results for each kernel
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        evaluation_results = {}

        for kernel, classifier in self.classifiers.items():
            start_time = time.time()
            predictions = classifier.predict(X_test)
            predict_time = time.time() - start_time

            accuracy = accuracy_score(y_test, predictions)

            evaluation_results[kernel] = {
                'accuracy': accuracy,
                'predict_time': predict_time,
                'predictions': predictions,
                'classification_report': classification_report(y_test, predictions, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, predictions)
            }

        return evaluation_results

    def get_best_kernel(self, metric: str = 'accuracy') -> Tuple[str, CSVMClassifier]:
        """
        Get the best performing kernel based on specified metric.

        Args:
            metric: Metric to use for comparison ('accuracy', 'fit_time', etc.)

        Returns:
            Tuple of (best_kernel_name, best_classifier)
        """
        if not self.results:
            raise ValueError("No results available. Run fit_all_kernels first.")

        if metric == 'accuracy':
            # Need evaluation results for accuracy
            raise ValueError("Run evaluate_all_kernels first to compare by accuracy")
        elif metric == 'fit_time':
            best_kernel = min(self.results.keys(),
                            key=lambda k: self.results[k]['fit_time'])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return best_kernel, self.classifiers[best_kernel]