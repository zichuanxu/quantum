"""
Unit tests for FileManager class.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from quantum.src.utils.file_manager import FileManager


class TestFileManager(unittest.TestCase):
    """Test cases for FileManager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_directories_exist(self):
        """Test directory creation."""
        self.file_manager.ensure_directories_exist()

        # Check that all directories were created
        self.assertTrue(self.file_manager.results_dir.exists())
        self.assertTrue(self.file_manager.images_dir.exists())
        self.assertTrue(self.file_manager.deutsch_images_dir.exists())
        self.assertTrue(self.file_manager.svm_images_dir.exists())

    def test_get_paths(self):
        """Test path generation methods."""
        results_path = self.file_manager.get_results_path("test.md")
        deutsch_path = self.file_manager.get_deutsch_image_path("test.png")
        svm_path = self.file_manager.get_svm_image_path("test.png")

        self.assertEqual(results_path.name, "test.md")
        self.assertEqual(deutsch_path.name, "test.png")
        self.assertEqual(svm_path.name, "test.png")
        self.assertIn("deutsch", str(deutsch_path))
        self.assertIn("svm", str(svm_path))

    def test_relative_image_paths(self):
        """Test relative path generation for markdown."""
        deutsch_rel = self.file_manager.get_relative_image_path("deutsch", "test.png")
        svm_rel = self.file_manager.get_relative_image_path("svm", "test.png")

        self.assertEqual(deutsch_rel, "images/deutsch/test.png")
        self.assertEqual(svm_rel, "images/svm/test.png")

        with self.assertRaises(ValueError):
            self.file_manager.get_relative_image_path("invalid", "test.png")

    def test_file_operations(self):
        """Test file existence and creation."""
        test_file = Path(self.temp_dir) / "test.txt"

        # File doesn't exist initially
        self.assertFalse(self.file_manager.file_exists(test_file))

        # Create file
        self.file_manager.create_file_if_not_exists(test_file, "test content")
        self.assertTrue(self.file_manager.file_exists(test_file))
        self.assertEqual(test_file.read_text(), "test content")

        # Don't overwrite existing file
        self.file_manager.create_file_if_not_exists(test_file, "new content")
        self.assertEqual(test_file.read_text(), "test content")


if __name__ == '__main__':
    unittest.main()