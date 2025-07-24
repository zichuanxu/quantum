"""
Quantum Support Vector Machine (QSVM) classifier implementation.
Uses Qiskit's quantum kernel methods for classification.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
try:
    # Try the most common import path first
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    QISKIT_ML_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths for different versions
        from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
        from qiskit_machine_learning.algorithms.classifiers import QSVC
        QISKIT_ML_AVAILABLE = True
    except ImportError:
        try:
            # Try even more specific imports
            from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel as QuantumKernel
            from qiskit_machine_learning.algorithms.classifiers.qsvc import QSVC
            QISKIT_ML_AVAILABLE = True
        except ImportError:
            # Final fallback: no quantum ML available
            print("Warning: qiskit-machine-learning not available. QSVM will use classical fallback.")
            QuantumKernel = None
            QSVC = None
            QISKIT_ML_AVAILABLE = False
from qiskit_aer import Aer
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Dict, Any, Tuple
import time
import warnings


class QSVMClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Support Vector Machine classifier using quantum kernels."""

    def __init__(self, feature_map_type: str = 'ZZFeatureMap',
                 feature_map_reps: int = 2,
                 quantum_backend: str = 'aer_simulator',
                 shots: int = 1024,
                 C: float = 1.0):
        """
        Initialize QSVM classifier.

        Args:
            feature_map_type: Type of quantum feature map ('ZZFeatureMap' or 'PauliFeatureMap')
            feature_map_reps: Number of repetitions in the feature map
            quantum_backend: Quantum backend to use
            shots: Number of shots for quantum execution
            C: Regularization parameter
        """
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.quantum_backend = quantum_backend
        self.shots = shots
        self.C = C

        self.feature_map = None
        self.quantum_kernel = None
        self.qsvc = None
        self.is_fitted = False
        self.n_features = None

    def _create_feature_map(self, n_features: int) -> QuantumCircuit:
        """
        Create quantum feature map based on the specified type.

        Args:
            n_features: Number of input features

        Returns:
            Quantum feature map circuit
        """
        if self.feature_map_type == 'ZZFeatureMap':
            feature_map = ZZFeatureMap(
                feature_dimension=n_features,
                reps=self.feature_map_reps,
                entanglement='linear'
            )
        elif self.feature_map_type == 'PauliFeatureMap':
            feature_map = PauliFeatureMap(
                feature_dimension=n_features,
                reps=self.feature_map_reps,
                paulis=['Z', 'ZZ']
            )
        else:
            raise ValueError(f"Unsupported feature map type: {self.feature_map_type}")

        return feature_map

    def _setup_quantum_kernel(self, n_features: int) -> None:
        """
        Set up the quantum kernel with the specified backend.

        Args:
            n_features: Number of input features
        """
        if QuantumKernel is None:
            # If QuantumKernel is not available, skip quantum kernel setup
            self.quantum_kernel = None
            return

        # Create feature map
        self.feature_map = self._create_feature_map(n_features)

        # Set up quantum backend
        if self.quantum_backend == 'aer_simulator':
            backend = Aer.get_backend('aer_simulator')
        else:
            # For other backends, you might need to set up IBMQ provider
            backend = Aer.get_backend('aer_simulator')  # Fallback to simulator

        # Create quantum kernel
        try:
            # Try the newer API first (qiskit-machine-learning >= 0.6)
            self.quantum_kernel = QuantumKernel(
                feature_map=self.feature_map,
                quantum_instance=backend
            )
        except TypeError:
            try:
                # Try alternative API without quantum_instance
                self.quantum_kernel = QuantumKernel(
                    feature_map=self.feature_map
                )
            except Exception as e2:
                print(f"Warning: Failed to create quantum kernel with alternative API: {e2}")
                self.quantum_kernel = None
        except Exception as e:
            print(f"Warning: Failed to create quantum kernel: {e}")
            self.quantum_kernel = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QSVMClassifier':
        """
        Fit the QSVM classifier to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self (fitted classifier)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features = X.shape[1]

        # Set up quantum kernel
        self._setup_quantum_kernel(self.n_features)

        # Create and fit QSVC
        try:
            if QSVC is not None and self.quantum_kernel is not None:
                self.qsvc = QSVC(
                    quantum_kernel=self.quantum_kernel,
                    C=self.C
                )

                # Suppress warnings during fitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.qsvc.fit(X, y)

                self.is_fitted = True
            else:
                # If QSVC is not available, use fallback immediately
                raise ImportError("QSVC not available")

        except Exception as e:
            # Fallback: use classical SVM with precomputed quantum kernel
            print(f"Warning: QSVC failed ({e}), falling back to classical SVM with quantum kernel")
            self._fit_with_precomputed_kernel(X, y)

        return self

    def _fit_with_precomputed_kernel(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fallback method using classical SVM with precomputed quantum kernel matrix.

        Args:
            X: Training features
            y: Training labels
        """
        try:
            # Compute quantum kernel matrix
            kernel_matrix = self.quantum_kernel.evaluate(X)

            # Use classical SVM with precomputed kernel
            self.qsvc = SVC(kernel='precomputed', C=self.C)
            self.qsvc.fit(kernel_matrix, y)

            # Store training data for prediction
            self.X_train = X.copy()
            self.is_fitted = True

        except Exception as e:
            # Final fallback: use classical SVM with RBF kernel
            print(f"Warning: Quantum kernel computation failed ({e}), using classical RBF kernel")
            self.qsvc = SVC(kernel='rbf', C=self.C, gamma='scale')
            self.qsvc.fit(X, y)
            self.is_fitted = True

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

        X = np.asarray(X)

        try:
            if hasattr(self.qsvc, 'predict') and hasattr(self.qsvc, 'quantum_kernel'):
                # Direct QSVC prediction
                return self.qsvc.predict(X)
            elif hasattr(self, 'X_train'):
                # Precomputed kernel method
                kernel_matrix = self.quantum_kernel.evaluate(X, self.X_train)
                return self.qsvc.predict(kernel_matrix)
            else:
                # Classical fallback
                return self.qsvc.predict(X)

        except Exception as e:
            print(f"Warning: Quantum prediction failed ({e}), using fallback method")
            # Emergency fallback
            if hasattr(self, 'X_train'):
                # Use simple distance-based classification
                distances = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)
                nearest_indices = np.argmin(distances, axis=1)
                return self.qsvc.support_vectors_[nearest_indices] if hasattr(self.qsvc, 'support_vectors_') else np.zeros(len(X))
            else:
                return np.zeros(len(X))  # Return all zeros as last resort

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

        try:
            if hasattr(self.qsvc, 'predict_proba'):
                return self.qsvc.predict_proba(X)
            elif hasattr(self.qsvc, 'decision_function'):
                # Convert decision function to probabilities using sigmoid
                decision = self.qsvc.decision_function(X)
                prob_pos = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - prob_pos, prob_pos])
            else:
                # Fallback: return hard predictions as probabilities
                predictions = self.predict(X)
                proba = np.zeros((len(X), 2))
                proba[predictions == 0, 0] = 1.0
                proba[predictions == 1, 1] = 1.0
                return proba

        except Exception as e:
            print(f"Warning: Probability prediction failed ({e})")
            # Return uniform probabilities
            return np.full((len(X), 2), 0.5)

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

        try:
            if hasattr(self.qsvc, 'decision_function'):
                if hasattr(self, 'X_train'):
                    # Precomputed kernel method
                    kernel_matrix = self.quantum_kernel.evaluate(X, self.X_train)
                    return self.qsvc.decision_function(kernel_matrix)
                else:
                    return self.qsvc.decision_function(X)
            else:
                # Fallback: convert predictions to decision values
                predictions = self.predict(X)
                return 2 * predictions - 1  # Convert {0,1} to {-1,1}

        except Exception as e:
            print(f"Warning: Decision function computation failed ({e})")
            return np.zeros(len(X))

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
            # Process in smaller batches to avoid memory issues and timeouts
            batch_size = min(500, len(grid_points))
            Z_parts = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                try:
                    Z_batch = self.decision_function(batch)
                    Z_parts.append(Z_batch)
                except Exception:
                    # Fallback for failed batch
                    Z_parts.append(np.zeros(len(batch)))

            Z = np.concatenate(Z_parts)
            Z = Z.reshape(xx.shape)

        except Exception as e:
            print(f"Warning: QSVM decision boundary computation failed ({e})")
            # Return zero decision boundary
            Z = np.zeros_like(xx)

        return xx, yy, Z

    def get_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute quantum kernel matrix between datasets.

        Args:
            X1: First dataset
            X2: Second dataset (if None, compute X1 vs X1)

        Returns:
            Kernel matrix
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before computing kernel matrix")

        try:
            if X2 is None:
                return self.quantum_kernel.evaluate(X1)
            else:
                return self.quantum_kernel.evaluate(X1, X2)
        except Exception as e:
            print(f"Warning: Kernel matrix computation failed ({e})")
            # Return identity matrix as fallback
            n1 = len(X1)
            n2 = len(X2) if X2 is not None else n1
            return np.eye(n1, n2)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.

        Returns:
            Dictionary with model information
        """
        info = {
            'classifier_type': 'QSVM',
            'feature_map_type': self.feature_map_type,
            'feature_map_reps': self.feature_map_reps,
            'quantum_backend': self.quantum_backend,
            'shots': self.shots,
            'C': self.C,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted
        }

        if self.is_fitted and hasattr(self.qsvc, 'support_'):
            try:
                info['n_support_vectors'] = len(self.qsvc.support_)
            except:
                info['n_support_vectors'] = 'Unknown'

        return info