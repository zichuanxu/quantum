"""
Main application entry point for Quantum Assignment 3.
Executes both Deutsch algorithm analysis and SVM comparison.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the quantum package to the path
sys.path.append(os.path.dirname(__file__))

from src.deutsch_algorithm.deutsch_workflow import DeutschWorkflow
from src.svm_comparison.svm_workflow import SVMWorkflow
from src.utils.file_manager import FileManager


def run_deutsch_analysis(base_dir: str = "quantum", shots: int = 1024) -> dict:
    """
    Run Deutsch algorithm analysis.

    Args:
        base_dir: Base directory for the project
        shots: Number of shots for quantum circuit execution

    Returns:
        Dictionary with analysis summary
    """
    print("Starting Deutsch Algorithm Analysis...")
    workflow = DeutschWorkflow(base_dir, shots)
    return workflow.run_complete_analysis()


def run_svm_comparison(base_dir: str = "quantum",
                      digit_pairs: list = None,
                      kernels: list = None,
                      random_state: int = 42) -> dict:
    """
    Run SVM comparison analysis.

    Args:
        base_dir: Base directory for the project
        digit_pairs: List of digit pairs to analyze
        kernels: List of kernels to test
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with analysis summary
    """
    if digit_pairs is None:
        digit_pairs = [(3, 4), (1, 2)]
    if kernels is None:
        kernels = ['linear', 'rbf']

    print("Starting SVM Comparison Analysis...")
    workflow = SVMWorkflow(base_dir, random_state)
    return workflow.run_complete_analysis(digit_pairs, kernels)


def run_complete_assignment(base_dir: str = "quantum",
                          shots: int = 1024,
                          digit_pairs: list = None,
                          kernels: list = None,
                          random_state: int = 42) -> dict:
    """
    Run the complete quantum assignment analysis.

    Args:
        base_dir: Base directory for the project
        shots: Number of shots for quantum circuits
        digit_pairs: List of digit pairs for SVM comparison
        kernels: List of kernels to test
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with complete analysis summary
    """
    print("=" * 80)
    print("QUANTUM COMPUTING ASSIGNMENT 3 - COMPLETE ANALYSIS")
    print("Deutsch Algorithm Implementation & Quantum vs Classical SVM Comparison")
    print("=" * 80)

    # Ensure base directory structure exists
    file_manager = FileManager(base_dir)
    file_manager.ensure_directories_exist()

    results = {
        'assignment_title': 'Quantum Computing Assignment 3',
        'components': ['Deutsch Algorithm', 'SVM Comparison'],
        'base_directory': base_dir
    }

    try:
        # Run Deutsch algorithm analysis
        print("\n" + "=" * 40)
        print("PART 1: DEUTSCH ALGORITHM ANALYSIS")
        print("=" * 40)

        deutsch_results = run_deutsch_analysis(base_dir, shots)
        results['deutsch_analysis'] = deutsch_results

        if 'error' not in deutsch_results:
            print(f"✓ Deutsch analysis completed successfully")
            print(f"  - Success rate: {deutsch_results['success_rate']:.1f}%")
            print(f"  - Report: {deutsch_results['report_path']}")
        else:
            print(f"✗ Deutsch analysis failed: {deutsch_results['error']}")

        # Run SVM comparison analysis
        print("\n" + "=" * 40)
        print("PART 2: SVM COMPARISON ANALYSIS")
        print("=" * 40)

        svm_results = run_svm_comparison(base_dir, digit_pairs, kernels, random_state)
        results['svm_comparison'] = svm_results

        if 'error' not in svm_results:
            print(f"✓ SVM comparison completed successfully")
            print(f"  - Success rate: {svm_results['success_rate']:.1f}%")
            print(f"  - Report: {svm_results['report_path']}")
        else:
            print(f"✗ SVM comparison failed: {svm_results['error']}")

        # Generate final summary
        print("\n" + "=" * 80)
        print("ASSIGNMENT COMPLETION SUMMARY")
        print("=" * 80)

        deutsch_success = 'error' not in deutsch_results
        svm_success = 'error' not in svm_results

        print(f"Deutsch Algorithm Analysis: {'✓ SUCCESS' if deutsch_success else '✗ FAILED'}")
        print(f"SVM Comparison Analysis: {'✓ SUCCESS' if svm_success else '✗ FAILED'}")

        if deutsch_success and svm_success:
            print("\n🎉 ASSIGNMENT COMPLETED SUCCESSFULLY! 🎉")
            print("\nGenerated Files:")
            print(f"  - Deutsch Report: {deutsch_results['report_path']}")
            print(f"  - SVM Report: {svm_results['report_path']}")
            print(f"  - Deutsch Images: {file_manager.deutsch_images_dir}")
            print(f"  - SVM Images: {file_manager.svm_images_dir}")

            results['status'] = 'SUCCESS'
            results['overall_success_rate'] = (
                (deutsch_results['success_rate'] + svm_results['success_rate']) / 2
            )
        else:
            print("\n⚠️  ASSIGNMENT PARTIALLY COMPLETED")
            results['status'] = 'PARTIAL'

        print("\nAll results have been saved to markdown files with organized visualizations.")
        print("You can now review the generated reports and submit your assignment.")

    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        results['status'] = 'FAILED'
        results['error'] = str(e)

    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Quantum Computing Assignment 3: Deutsch Algorithm & SVM Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run complete assignment
  python main.py --deutsch-only            # Run only Deutsch algorithm
  python main.py --svm-only                # Run only SVM comparison
  python main.py --shots 2048              # Use 2048 shots for quantum circuits
  python main.py --digits 0 1 5 6         # Compare digits 0vs1 and 5vs6
  python main.py --kernels linear rbf poly # Test linear, RBF, and polynomial kernels
        """
    )

    parser.add_argument('--deutsch-only', action='store_true',
                       help='Run only Deutsch algorithm analysis')
    parser.add_argument('--svm-only', action='store_true',
                       help='Run only SVM comparison analysis')
    parser.add_argument('--base-dir', type=str, default='quantum',
                       help='Base directory for the project (default: quantum)')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of shots for quantum circuits (default: 1024)')
    parser.add_argument('--digits', type=int, nargs='+',
                       help='Digits to compare (provide 4 numbers for 2 pairs, e.g., 3 4 1 2)')
    parser.add_argument('--kernels', type=str, nargs='+',
                       choices=['linear', 'rbf', 'poly', 'sigmoid'],
                       default=['linear', 'rbf'],
                       help='Kernels to test for SVM (default: linear rbf)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Process digit pairs
    digit_pairs = [(3, 4), (1, 2)]  # Default
    if args.digits:
        if len(args.digits) % 2 != 0:
            print("Error: Number of digits must be even (pairs of digits)")
            sys.exit(1)
        digit_pairs = [(args.digits[i], args.digits[i+1])
                      for i in range(0, len(args.digits), 2)]

    # Run requested analysis
    if args.deutsch_only:
        results = run_deutsch_analysis(args.base_dir, args.shots)
    elif args.svm_only:
        results = run_svm_comparison(args.base_dir, digit_pairs, args.kernels, args.random_state)
    else:
        results = run_complete_assignment(
            args.base_dir, args.shots, digit_pairs, args.kernels, args.random_state
        )

    # Exit with appropriate code
    if results.get('status') == 'SUCCESS':
        sys.exit(0)
    elif results.get('status') == 'PARTIAL':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()