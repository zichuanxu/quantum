"""
File management utilities for quantum assignment.
Handles directory creation and path management.
"""

import os
from pathlib import Path
from typing import Union


class FileManager:
    """Manages file operations and directory structure for the quantum assignment."""

    def __init__(self, base_dir: str = "quantum"):
        """Initialize FileManager with base directory."""
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.images_dir = self.results_dir / "images"
        self.deutsch_images_dir = self.images_dir / "deutsch"
        self.svm_images_dir = self.images_dir / "svm"

    def ensure_directories_exist(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.results_dir,
            self.images_dir,
            self.deutsch_images_dir,
            self.svm_images_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_results_path(self, filename: str) -> Path:
        """Get full path for a results file."""
        return self.results_dir / filename

    def get_deutsch_image_path(self, filename: str) -> Path:
        """Get full path for a Deutsch algorithm image."""
        return self.deutsch_images_dir / filename

    def get_svm_image_path(self, filename: str) -> Path:
        """Get full path for an SVM comparison image."""
        return self.svm_images_dir / filename

    def get_relative_image_path(self, image_type: str, filename: str) -> str:
        """Get relative path for markdown image references."""
        if image_type == "deutsch":
            return f"images/deutsch/{filename}"
        elif image_type == "svm":
            return f"images/svm/{filename}"
        else:
            raise ValueError(f"Unknown image type: {image_type}")

    def file_exists(self, filepath: Union[str, Path]) -> bool:
        """Check if a file exists."""
        return Path(filepath).exists()

    def create_file_if_not_exists(self, filepath: Union[str, Path], content: str = "") -> None:
        """Create a file with content if it doesn't exist."""
        path = Path(filepath)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)