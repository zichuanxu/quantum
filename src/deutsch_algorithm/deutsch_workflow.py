"""
Main execution workflow for Deutsch algorithm implementation.
Orchestrates circuit creation, state analysis, visualization, and reporting.
"""

from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add the quantum package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.deutsch_algorithm.circuit_builder import CircuitBuilder
from src.deutsch_algorithm.state_analyzer import StateAnalyzer
from src.deutsch_algorithm.visualizer import Visualizer
from src.utils.file_manager import FileManager
from src.utils.markdown_generator import DeutschReportGenerator


class DeutschWorkflow:
    """Main workflow for executing Deutsch algorithm analysis."""

    def __init__(self, base_dir: str = ".", shots: int = 1024):
        """
        Initialize Deutsch workflow.

        Args:
            base_dir: Base directory for the project
            shots: Number of shots for quantum circuit execution
        """
        self.base_dir = base_dir
        self.shots = shots

        # Initialize components
        self.file_manager = FileManager(base_dir)
        self.circuit_builder = CircuitBuilder(shots)
        self.state_analyzer = StateAnalyzer()
        self.visualizer = Visualizer()
        self.report_generator = DeutschReportGenerator()

        # Ensure directories exist
        self.file_manager.ensure_directories_exist()

        # Storage for results
        self.all_results = {}

    def execute_single_case(self, case_number: int) -> Dict[str, Any]:
        """
        Execute analysis for a single Deutsch algorithm case.

        Args:
            case_number: Case number (1-4)

        Returns:
            Dictionary with complete analysis results
        """
        print(f"Executing Deutsch Algorithm Case {case_number}...")

        # Get circuit description
        circuit_description = self.circuit_builder.get_circuit_description(case_number)

        # Create circuit
        if case_number == 1:
            circuit = self.circuit_builder.create_circuit_case_1()
        elif case_number == 2:
            circuit = self.circuit_builder.create_circuit_case_2()
        elif case_number == 3:
            circuit = self.circuit_builder.create_circuit_case_3()
        elif case_number == 4:
            circuit = self.circuit_builder.create_circuit_case_4()
        else:
            raise ValueError(f"Invalid case number: {case_number}")

        # Execute circuit
        execution_results = self.circuit_builder.execute_circuit(circuit)

        # Analyze quantum states
        analysis_results = self.state_analyzer.analyze_complete_evolution(circuit)

        # Create visualizations
        visualization_paths = self.visualizer.create_all_visualizations(
            case_number, circuit, analysis_results,
            self.file_manager.deutsch_images_dir
        )

        # Compile complete results
        complete_results = {
            'case_number': case_number,
            'circuit_description': circuit_description,
            'circuit': circuit,
            'execution_results': execution_results,
            'analysis_results': analysis_results,
            'visualization_paths': visualization_paths
        }

        return complete_results

    def execute_all_cases(self) -> Dict[int, Dict[str, Any]]:
        """
        Execute analysis for all four Deutsch algorithm cases.

        Returns:
            Dictionary mapping case numbers to results
        """
        print("Starting Deutsch Algorithm Analysis for all cases...")

        for case_number in range(1, 5):
            try:
                self.all_results[case_number] = self.execute_single_case(case_number)
                print(f"✓ Case {case_number} completed successfully")
            except Exception as e:
                print(f"✗ Case {case_number} failed: {e}")
                self.all_results[case_number] = {
                    'case_number': case_number,
                    'error': str(e)
                }

        return self.all_results

    def create_comparison_visualizations(self) -> Dict[str, str]:
        """
        Create comparison visualizations across all cases.

        Returns:
            Dictionary mapping visualization types to file paths
        """
        print("Creating comparison visualizations...")

        comparison_paths = {}

        try:
            # Extract execution results for comparison
            execution_results = {}
            for case_num, results in self.all_results.items():
                if 'execution_results' in results:
                    execution_results[case_num] = results['execution_results']

            if execution_results:
                # Create measurement results comparison
                comparison_fig = self.visualizer.create_measurement_results_plot(execution_results)
                comparison_path = self.visualizer.save_visualization(
                    comparison_fig, "measurement_comparison",
                    self.file_manager.deutsch_images_dir
                )
                comparison_paths['measurement_comparison'] = comparison_path

        except Exception as e:
            print(f"Warning: Comparison visualization failed: {e}")

        return comparison_paths

    def generate_markdown_report(self) -> str:
        """
        Generate comprehensive markdown report for all cases.

        Returns:
            Path to the generated markdown report
        """
        print("Generating markdown report...")

        # Initialize report
        self.report_generator.create_deutsch_report_template()

        # Add analysis for each case
        for case_number in range(1, 5):
            if case_number in self.all_results and 'error' not in self.all_results[case_number]:
                results = self.all_results[case_number]

                # Extract data for report
                circuit_desc = results['circuit_description']
                analysis = results['analysis_results']
                viz_paths = results['visualization_paths']
                execution = results['execution_results']

                # Convert states to string format for report
                states_before = {
                    'q0': analysis['states_before_hadamard']['q0']['state_string'],
                    'q1': analysis['states_before_hadamard']['q1']['state_string']
                }

                states_after = {
                    'q0': analysis['states_after_hadamard']['q0']['state_string'],
                    'q1': analysis['states_after_hadamard']['q1']['state_string']
                }

                # Get relative paths for markdown
                circuit_image = self.file_manager.get_relative_image_path(
                    "deutsch", Path(viz_paths['circuit']).name
                )

                bloch_images = []
                for bloch_path in viz_paths['bloch_spheres']:
                    relative_path = self.file_manager.get_relative_image_path(
                        "deutsch", Path(bloch_path).name
                    )
                    bloch_images.append(relative_path)

                # Add case analysis to report
                self.report_generator.add_circuit_case_analysis(
                    case_number=case_number,
                    circuit_description=circuit_desc['name'],
                    oracle_type=circuit_desc['oracle_type'],
                    states_before=states_before,
                    states_after=states_after,
                    measurement_results=execution['counts'],
                    circuit_image_path=circuit_image,
                    bloch_images=bloch_images
                )
            else:
                # Add error information
                error_msg = self.all_results.get(case_number, {}).get('error', 'Unknown error')
                self.report_generator.add_header(f"Case {case_number}: Error", 2)
                self.report_generator.add_paragraph(f"Analysis failed: {error_msg}")

        # Add comparison section if available
        comparison_paths = self.create_comparison_visualizations()
        if comparison_paths:
            self.report_generator.add_header("Cross-Case Comparison", 2)
            self.report_generator.add_paragraph(
                "The following visualization compares measurement results across all circuit cases:"
            )

            if 'measurement_comparison' in comparison_paths:
                comparison_image = self.file_manager.get_relative_image_path(
                    "deutsch", Path(comparison_paths['measurement_comparison']).name
                )
                self.report_generator.add_image(
                    "Measurement Results Comparison", comparison_image,
                    "Comparison of measurement probabilities and counts across all four cases"
                )

        # Add theoretical analysis
        self.report_generator.add_header("Theoretical Analysis", 2)
        self.report_generator.add_paragraph(
            "The Deutsch algorithm demonstrates quantum parallelism by determining whether "
            "a function is constant or balanced with just one quantum query, compared to "
            "two classical queries required in the worst case."
        )

        self.report_generator.add_paragraph(
            "**Key Observations:**"
        )

        observations = [
            "Cases 1 and 3 implement constant functions (f(x) = 0 and f(x) = 1 respectively)",
            "Cases 2 and 4 implement balanced functions (f(x) = NOT x and f(x) = x respectively)",
            "The final measurement of q0 reveals the function type: 0 for constant, 1 for balanced",
            "Constant functions: Case 1 (f(x) = 0) and Case 3 (f(x) = 1) both measure 0",
            "Balanced functions: Case 2 (f(x) = NOT x) and Case 4 (f(x) = x) both measure 1",
            "Quantum superposition allows simultaneous evaluation of the function on all inputs"
        ]

        self.report_generator.add_list(observations)

        # Save report
        report_path = self.file_manager.get_results_path("deutsch_results.md")
        self.report_generator.save_to_file(report_path)

        print(f"✓ Report saved to: {report_path}")
        return str(report_path)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete Deutsch algorithm analysis workflow.

        Returns:
            Dictionary with summary of all results and paths
        """
        print("=" * 60)
        print("DEUTSCH ALGORITHM ANALYSIS - COMPLETE WORKFLOW")
        print("=" * 60)

        try:
            # Execute all cases
            results = self.execute_all_cases()

            # Generate report
            report_path = self.generate_markdown_report()

            # Summary
            successful_cases = [k for k, v in results.items() if 'error' not in v]
            failed_cases = [k for k, v in results.items() if 'error' in v]

            summary = {
                'total_cases': 4,
                'successful_cases': len(successful_cases),
                'failed_cases': len(failed_cases),
                'success_rate': len(successful_cases) / 4 * 100,
                'report_path': report_path,
                'results': results
            }

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Successful cases: {successful_cases}")
            if failed_cases:
                print(f"Failed cases: {failed_cases}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            print(f"Report generated: {report_path}")
            print(f"Images saved to: {self.file_manager.deutsch_images_dir}")

            return summary

        except Exception as e:
            print(f"✗ Complete analysis failed: {e}")
            return {
                'error': str(e),
                'total_cases': 4,
                'successful_cases': 0,
                'failed_cases': 4,
                'success_rate': 0.0
            }


def main():
    """Main function to run Deutsch algorithm analysis."""
    workflow = DeutschWorkflow()
    summary = workflow.run_complete_analysis()
    return summary


if __name__ == "__main__":
    main()