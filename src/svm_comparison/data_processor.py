"""
Data processing module for SVM comparison.
Handles dataset loading, preprocessing, and dimensionality reduction.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


class DataProcessor:
    """Processes data for quantum and classical SVM comparison."""

    def __init__(self, random_state: int = 42):
        """
        Initialize DataProcessor with random state for reproducibility.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.original_data = None
        self.original_labels = None

    def load_digits_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the digits dataset from sklearn.

        Returns:
            Tuple of (data, labels) arrays
        """
        digits = load_digits()
        self.original_data = digits.data
        self.original_labels = digits.target

        return self.original_data, self.original_labels

    def select_digit_pairs(self, data: np.ndarray, labels: np.ndarray,
                          digit1: int, digit2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract data for two specific digit classes.

        Args:
            data: Full dataset
            labels: Full label array
            digit1: First digit class to extract
            digit2: Second digit class to extract

        Returns:
            Tuple of (filtered_data, binary_labels)
        """
        # Find indices for the two digit classes
        mask = (labels == digit1) | (labels == digit2)
        filtered_data = data[mask]
        filtered_labels = labels[mask]

        # Convert to binary labels (0 for digit1, 1 for digit2)
        binary_labels = np.where(filtered_labels == digit1, 0, 1)

        return filtered_data, binary_labels

    def reduce_dimensionality(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA dimensionality reduction to the data.

        Args:
            data: Input data array
            n_components: Number of principal components to keep

        Returns:
            Reduced dimensionality data
        """
        # Standardize the data first
        data_scaled = self.scaler.fit_transform(data)

        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        data_reduced = self.pca.fit_transform(data_scaled)

        return data_reduced

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            X: Feature data
            y: Labels
            test_size: Fraction of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size,
                              random_state=self.random_state, stratify=y)

    def get_dataset_info(self, X: np.ndarray, y: np.ndarray,
                        digit1: int, digit2: int) -> Dict[str, Any]:
        """
        Get information about the processed dataset.

        Args:
            X: Feature data
            y: Labels
            digit1: First digit class
            digit2: Second digit class

        Returns:
            Dictionary with dataset information
        """
        unique_labels, counts = np.unique(y, return_counts=True)

        info = {
            'digit_pair': (digit1, digit2),
            'total_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': dict(zip(unique_labels, counts)),
            'class_balance': counts[0] / counts[1] if len(counts) == 2 else 1.0
        }

        if self.pca is not None:
            info['explained_variance_ratio'] = self.pca.explained_variance_ratio_
            info['total_explained_variance'] = np.sum(self.pca.explained_variance_ratio_)

        return info

    def create_visualization_grid(self, X: np.ndarray, y: np.ndarray,
                                digit1: int, digit2: int,
                                n_samples: int = 10) -> plt.Figure:
        """
        Create a visualization grid showing sample digits from each class.

        Args:
            X: Original high-dimensional data (64 features)
            y: Binary labels
            digit1: First digit class
            digit2: Second digit class
            n_samples: Number of samples to show per class

        Returns:
            Matplotlib figure with sample digits
        """
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))
        fig.suptitle(f'Sample Digits: {digit1} vs {digit2}', fontsize=14)

        # Get indices for each class
        class0_indices = np.where(y == 0)[0][:n_samples]
        class1_indices = np.where(y == 1)[0][:n_samples]

        # Plot samples from class 0 (digit1)
        for i, idx in enumerate(class0_indices):
            if i < n_samples:
                axes[0, i].imshow(X[idx].reshape(8, 8), cmap='gray')
                axes[0, i].set_title(f'Digit {digit1}')
                axes[0, i].axis('off')

        # Plot samples from class 1 (digit2)
        for i, idx in enumerate(class1_indices):
            if i < n_samples:
                axes[1, i].imshow(X[idx].reshape(8, 8), cmap='gray')
                axes[1, i].set_title(f'Digit {digit2}')
                axes[1, i].axis('off')

        # Hide unused subplots
        for i in range(len(class0_indices), n_samples):
            axes[0, i].axis('off')
        for i in range(len(class1_indices), n_samples):
            axes[1, i].axis('off')

        plt.tight_layout()
        return fig

    def create_pca_visualization(self, X_reduced: np.ndarray, y: np.ndarray,
                               digit1: int, digit2: int) -> plt.Figure:
        """
        Create a scatter plot of the PCA-reduced data.

        Args:
            X_reduced: PCA-reduced data (2D)
            y: Binary labels
            digit1: First digit class
            digit2: Second digit class

        Returns:
            Matplotlib figure with PCA visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create scatter plot
        colors = ['blue', 'red']
        labels = [f'Digit {digit1}', f'Digit {digit2}']

        for class_label in [0, 1]:
            mask = y == class_label
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                      c=colors[class_label], label=labels[class_label],
                      alpha=0.7, s=50)

        ax.set_xlabel(f'First Principal Component')
        ax.set_ylabel(f'Second Principal Component')
        ax.set_title(f'PCA Visualization: Digits {digit1} vs {digit2}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add explained variance information
        if self.pca is not None:
            var_text = f'Explained Variance: PC1={self.pca.explained_variance_ratio_[0]:.3f}, '
            var_text += f'PC2={self.pca.explained_variance_ratio_[1]:.3f}'
            ax.text(0.02, 0.98, var_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def process_complete_dataset(self, digit_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Process multiple digit pairs for comparison experiments.

        Args:
            digit_pairs: List of (digit1, digit2) tuples to process

        Returns:
            Dictionary mapping digit pairs to processed data
        """
        if self.original_data is None:
            self.load_digits_data()

        processed_datasets = {}

        for digit1, digit2 in digit_pairs:
            # Select digit pair
            X_pair, y_pair = self.select_digit_pairs(self.original_data, self.original_labels,
                                                   digit1, digit2)

            # Reduce dimensionality
            X_reduced = self.reduce_dimensionality(X_pair, n_components=2)

            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X_reduced, y_pair)

            # Get dataset info
            dataset_info = self.get_dataset_info(X_reduced, y_pair, digit1, digit2)

            processed_datasets[(digit1, digit2)] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_full': X_reduced,
                'y_full': y_pair,
                'X_original': X_pair,
                'info': dataset_info
            }

        return processed_datasets

    def get_feature_range(self, X: np.ndarray, padding: float = 0.1) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the range of features for plotting decision boundaries.

        Args:
            X: Feature data (2D)
            padding: Padding factor for the range

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max))
        """
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        return (x_min, x_max), (y_min, y_max)

    def create_mesh_grid(self, X: np.ndarray, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a mesh grid for decision boundary plotting.

        Args:
            X: Feature data to determine range
            resolution: Number of points per dimension

        Returns:
            Tuple of (xx, yy) mesh grid arrays
        """
        (x_min, x_max), (y_min, y_max) = self.get_feature_range(X)

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))

        return xx, yy