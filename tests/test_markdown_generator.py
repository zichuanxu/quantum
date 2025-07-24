"""
Unit tests for MarkdownGenerator classes.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from quantum.src.utils.markdown_generator import MarkdownGenerator, DeutschReportGenerator, SVMReportGenerator


class TestMarkdownGenerator(unittest.TestCase):
    """Test cases for MarkdownGenerator functionality."""

    def setUp(self):
        """Set up test environment."""
        self.generator = MarkdownGenerator()

    def test_add_header(self):
        """Test header addition."""
        self.generator.add_header("Test Header", 1)
        self.generator.add_header("Sub Header", 2)

        content = self.generator.get_content()
        self.assertIn("# Test Header", content)
        self.assertIn("## Sub Header", content)

        with self.assertRaises(ValueError):
            self.generator.add_header("Invalid", 7)

    def test_add_paragraph(self):
        """Test paragraph addition."""
        self.generator.add_paragraph("Test paragraph content.")
        content = self.generator.get_content()
        self.assertIn("Test paragraph content.", content)

    def test_add_image(self):
        """Test image addition."""
        self.generator.add_image("Alt text", "path/to/image.png", "Caption text")
        content = self.generator.get_content()
        self.assertIn("![Alt text](path/to/image.png)", content)
        self.assertIn("*Caption text*", content)

    def test_add_table(self):
        """Test table addition."""
        headers = ["Col1", "Col2", "Col3"]
        rows = [["A", "B", "C"], ["1", "2", "3"]]
        self.generator.add_table(headers, rows)

        content = self.generator.get_content()
        self.assertIn("| Col1 | Col2 | Col3 |", content)
        self.assertIn("| --- | --- | --- |", content)
        self.assertIn("| A | B | C |", content)

        with self.assertRaises(ValueError):
            self.generator.add_table(headers, [["A", "B"]])  # Wrong row length

    def test_add_list(self):
        """Test list addition."""
        items = ["Item 1", "Item 2", "Item 3"]

        # Unordered list
        self.generator.add_list(items, ordered=False)
        content = self.generator.get_content()
        self.assertIn("- Item 1", content)

        self.generator.clear()

        # Ordered list
        self.generator.add_list(items, ordered=True)
        content = self.generator.get_content()
        self.assertIn("1. Item 1", content)
        self.assertIn("2. Item 2", content)

    def test_clear_and_save(self):
        """Test content clearing and file saving."""
        self.generator.add_header("Test")
        self.assertNotEqual(self.generator.get_content(), "")

        self.generator.clear()
        self.assertEqual(self.generator.get_content(), "")

        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test.md"
            self.generator.add_paragraph("Test content")
            self.generator.save_to_file(filepath)

            self.assertTrue(filepath.exists())
            self.assertIn("Test content", filepath.read_text())


class TestDeutschReportGenerator(unittest.TestCase):
    """Test cases for DeutschReportGenerator."""

    def setUp(self):
        """Set up test environment."""
        self.generator = DeutschReportGenerator()

    def test_create_template(self):
        """Test Deutsch report template creation."""
        self.generator.create_deutsch_report_template()
        content = self.generator.get_content()

        self.assertIn("Deutsch Algorithm Implementation Results", content)
        self.assertIn("Overview", content)
        self.assertIn("Generated on:", content)

    def test_add_circuit_case_analysis(self):
        """Test circuit case analysis addition."""
        states_before = {"q0": "|0⟩", "q1": "|1⟩"}
        states_after = {"q0": "|+⟩", "q1": "|1⟩"}
        measurement_results = {"0": 512, "1": 512}

        self.generator.add_circuit_case_analysis(
            case_number=1,
            circuit_description="Identity Function",
            oracle_type="No oracle gates",
            states_before=states_before,
            states_after=states_after,
            measurement_results=measurement_results,
            circuit_image_path="images/circuit1.png",
            bloch_images=["images/bloch1.png", "images/bloch2.png"]
        )

        content = self.generator.get_content()
        self.assertIn("Case 1: Identity Function", content)
        self.assertIn("Oracle Type: No oracle gates", content)
        self.assertIn("q0: |0⟩", content)
        self.assertIn("Outcome '0': 512/1024 (50.0%)", content)


class TestSVMReportGenerator(unittest.TestCase):
    """Test cases for SVMReportGenerator."""

    def setUp(self):
        """Set up test environment."""
        self.generator = SVMReportGenerator()

    def test_create_template(self):
        """Test SVM report template creation."""
        self.generator.create_svm_report_template()
        content = self.generator.get_content()

        self.assertIn("Quantum SVM vs Classical SVM Comparison", content)
        self.assertIn("Overview", content)
        self.assertIn("Generated on:", content)

    def test_add_experiment_results(self):
        """Test experiment results addition."""
        qsvm_results = {"training_time": 2.5, "accuracy": 85.5}
        csvm_results = {"training_time": 0.1, "accuracy": 87.2}

        self.generator.add_experiment_results(
            digit_pair=(3, 4),
            kernel_type="linear",
            qsvm_results=qsvm_results,
            csvm_results=csvm_results,
            decision_boundary_image="images/boundaries.png"
        )

        content = self.generator.get_content()
        self.assertIn("Digits 3 vs 4", content)
        self.assertIn("2.5000", content)  # QSVM time
        self.assertIn("0.1000", content)  # CSVM time
        self.assertIn("85.50", content)   # QSVM accuracy


if __name__ == '__main__':
    unittest.main()