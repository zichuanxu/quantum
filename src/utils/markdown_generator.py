"""
Markdown report generation utilities for quantum assignment.
Creates structured reports with proper image references.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime


class MarkdownGenerator:
    """Generates markdown reports for quantum computing experiments."""

    def __init__(self):
        """Initialize MarkdownGenerator."""
        self.content = []

    def add_header(self, text: str, level: int = 1) -> None:
        """Add a header to the markdown content."""
        if level < 1 or level > 6:
            raise ValueError("Header level must be between 1 and 6")

        header_prefix = "#" * level
        self.content.append(f"{header_prefix} {text}\n")

    def add_paragraph(self, text: str) -> None:
        """Add a paragraph to the markdown content."""
        self.content.append(f"{text}\n")

    def add_image(self, alt_text: str, image_path: str, caption: Optional[str] = None) -> None:
        """Add an image reference to the markdown content."""
        self.content.append(f"![{alt_text}]({image_path})")
        if caption:
            self.content.append(f"\n*{caption}*\n")
        else:
            self.content.append("")

    def add_code_block(self, code: str, language: str = "") -> None:
        """Add a code block to the markdown content."""
        self.content.append(f"```{language}")
        self.content.append(code)
        self.content.append("```\n")

    def add_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Add a table to the markdown content."""
        if not headers or not rows:
            return

        # Header row
        header_row = "| " + " | ".join(headers) + " |"
        self.content.append(header_row)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        self.content.append(separator)

        # Data rows
        for row in rows:
            if len(row) != len(headers):
                raise ValueError("Row length must match header length")
            data_row = "| " + " | ".join(row) + " |"
            self.content.append(data_row)

        self.content.append("")

    def add_list(self, items: List[str], ordered: bool = False) -> None:
        """Add a list to the markdown content."""
        for i, item in enumerate(items):
            if ordered:
                self.content.append(f"{i+1}. {item}")
            else:
                self.content.append(f"- {item}")
        self.content.append("")

    def add_horizontal_rule(self) -> None:
        """Add a horizontal rule to the markdown content."""
        self.content.append("---\n")

    def clear(self) -> None:
        """Clear all content."""
        self.content = []

    def get_content(self) -> str:
        """Get the complete markdown content as a string."""
        return "\n".join(self.content)

    def save_to_file(self, filepath: Path) -> None:
        """Save the markdown content to a file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.get_content(), encoding='utf-8')


class DeutschReportGenerator(MarkdownGenerator):
    """Specialized markdown generator for Deutsch algorithm reports."""

    def create_deutsch_report_template(self) -> None:
        """Create the basic template for Deutsch algorithm report."""
        self.clear()
        self.add_header("Deutsch Algorithm Implementation Results")
        self.add_paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_horizontal_rule()

        self.add_header("Overview", 2)
        self.add_paragraph(
            "This report presents the implementation and analysis of the Deutsch algorithm "
            "across four different quantum circuit configurations. Each case demonstrates "
            "different oracle functions and their impact on quantum state evolution."
        )

    def add_circuit_case_analysis(self, case_number: int, circuit_description: str,
                                oracle_type: str, states_before: Dict[str, Any],
                                states_after: Dict[str, Any], measurement_results: Dict[str, int],
                                circuit_image_path: str, bloch_images: List[str]) -> None:
        """Add analysis for a specific circuit case."""
        self.add_header(f"Case {case_number}: {circuit_description}", 2)

        self.add_header("Circuit Description", 3)
        self.add_paragraph(f"Oracle Type: {oracle_type}")
        self.add_image(f"Circuit Case {case_number}", circuit_image_path,
                      f"Quantum circuit for Case {case_number}")

        self.add_header("Quantum State Analysis", 3)
        self.add_paragraph("**States before final Hadamard gate:**")
        self.add_paragraph(f"- q0: {states_before.get('q0', 'N/A')}")
        self.add_paragraph(f"- q1: {states_before.get('q1', 'N/A')}")

        self.add_paragraph("**States after final Hadamard gate:**")
        self.add_paragraph(f"- q0: {states_after.get('q0', 'N/A')}")
        self.add_paragraph(f"- q1: {states_after.get('q1', 'N/A')}")

        self.add_header("Bloch Sphere Visualizations", 3)
        for i, bloch_path in enumerate(bloch_images):
            self.add_image(f"Bloch Sphere {i+1}", bloch_path,
                          f"Bloch sphere representation for qubit {i}")

        self.add_header("Measurement Results", 3)
        total_shots = sum(measurement_results.values())
        for outcome, count in measurement_results.items():
            probability = count / total_shots * 100
            self.add_paragraph(f"- Outcome '{outcome}': {count}/{total_shots} ({probability:.1f}%)")


class SVMReportGenerator(MarkdownGenerator):
    """Specialized markdown generator for SVM comparison reports."""

    def create_svm_report_template(self) -> None:
        """Create the basic template for SVM comparison report."""
        self.clear()
        self.add_header("Quantum SVM vs Classical SVM Comparison")
        self.add_paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_horizontal_rule()

        self.add_header("Overview", 2)
        self.add_paragraph(
            "This report compares the performance of Quantum Support Vector Machine (QSVM) "
            "and Classical Support Vector Machine (CSVM) on digit classification tasks. "
            "The comparison includes execution time, accuracy, and decision boundary analysis."
        )

    def add_experiment_results(self, digit_pair: tuple, kernel_type: str,
                             qsvm_results: Dict[str, Any], csvm_results: Dict[str, Any],
                             decision_boundary_image: str) -> None:
        """Add results for a specific experiment."""
        self.add_header(f"Experiment: Digits {digit_pair[0]} vs {digit_pair[1]} ({kernel_type} kernel)", 2)

        # Performance comparison table
        self.add_header("Performance Comparison", 3)
        headers = ["Metric", "QSVM", "CSVM", "Difference"]
        rows = [
            ["Training Time (s)", f"{qsvm_results['training_time']:.4f}",
             f"{csvm_results['training_time']:.4f}",
             f"{qsvm_results['training_time'] - csvm_results['training_time']:.4f}"],
            ["Accuracy (%)", f"{qsvm_results['accuracy']:.2f}",
             f"{csvm_results['accuracy']:.2f}",
             f"{qsvm_results['accuracy'] - csvm_results['accuracy']:.2f}"]
        ]
        self.add_table(headers, rows)

        # Decision boundary visualization
        self.add_header("Decision Boundary Visualization", 3)
        self.add_image("Decision Boundaries", decision_boundary_image,
                      f"Decision boundaries for digits {digit_pair[0]} vs {digit_pair[1]}")

        # Analysis
        self.add_header("Analysis", 3)
        time_ratio = qsvm_results['training_time'] / csvm_results['training_time']
        if time_ratio > 1:
            self.add_paragraph(f"QSVM took {time_ratio:.1f}x longer to train than CSVM.")
        else:
            self.add_paragraph(f"QSVM was {1/time_ratio:.1f}x faster to train than CSVM.")

        accuracy_diff = qsvm_results['accuracy'] - csvm_results['accuracy']
        if accuracy_diff > 0:
            self.add_paragraph(f"QSVM achieved {accuracy_diff:.2f}% higher accuracy than CSVM.")
        else:
            self.add_paragraph(f"CSVM achieved {abs(accuracy_diff):.2f}% higher accuracy than QSVM.")

    def add_summary_analysis(self, all_results: List[Dict[str, Any]]) -> None:
        """Add overall summary analysis."""
        self.add_header("Summary Analysis", 2)

        # Calculate averages
        avg_qsvm_time = sum(r['qsvm_time'] for r in all_results) / len(all_results)
        avg_csvm_time = sum(r['csvm_time'] for r in all_results) / len(all_results)
        avg_qsvm_acc = sum(r['qsvm_accuracy'] for r in all_results) / len(all_results)
        avg_csvm_acc = sum(r['csvm_accuracy'] for r in all_results) / len(all_results)

        self.add_paragraph("**Overall Performance Summary:**")
        self.add_paragraph(f"- Average QSVM training time: {avg_qsvm_time:.4f}s")
        self.add_paragraph(f"- Average CSVM training time: {avg_csvm_time:.4f}s")
        self.add_paragraph(f"- Average QSVM accuracy: {avg_qsvm_acc:.2f}%")
        self.add_paragraph(f"- Average CSVM accuracy: {avg_csvm_acc:.2f}%")

        self.add_paragraph("**Key Findings:**")
        if avg_qsvm_time > avg_csvm_time:
            self.add_paragraph("- QSVM generally requires more training time than CSVM")
        else:
            self.add_paragraph("- QSVM shows competitive training times compared to CSVM")

        if avg_qsvm_acc > avg_csvm_acc:
            self.add_paragraph("- QSVM demonstrates superior classification accuracy")
        else:
            self.add_paragraph("- CSVM maintains competitive or superior accuracy performance")