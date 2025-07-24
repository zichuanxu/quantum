"""
Visualization module for Deutsch algorithm implementation.
Creates Bloch sphere plots, circuit diagrams, and comparison visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_vector, circuit_drawer
from qiskit.quantum_info import DensityMatrix
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """Creates visualizations for Deutsch algorithm analysis."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 300):
        """
        Initialize Visualizer with display parameters.

        Args:
            figsize: Figure size for plots
            dpi: Resolution for saved images
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('default')

    def create_bloch_sphere(self, bloch_coordinates: Tuple[float, float, float],
                          title: str = "Quantum State",
                          qubit_label: str = "q") -> plt.Figure:
        """
        Create a Bloch sphere visualization for a quantum state.

        Args:
            bloch_coordinates: (x, y, z) coordinates on Bloch sphere
            title: Title for the plot
            qubit_label: Label for the qubit

        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        try:
            # Use Qiskit's built-in Bloch sphere plotting
            ax = fig.add_subplot(111, projection='3d')

            # Draw the Bloch sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot the sphere with transparency
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')

            # Draw coordinate axes
            ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
            ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
            ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)

            # Draw equatorial and meridian circles
            circle_u = np.linspace(0, 2 * np.pi, 100)
            # Equatorial circle (z=0)
            ax.plot(np.cos(circle_u), np.sin(circle_u), 0, 'k-', alpha=0.3)
            # Meridian circle (y=0)
            ax.plot(np.cos(circle_u), 0, np.sin(circle_u), 'k-', alpha=0.3)
            # Meridian circle (x=0)
            ax.plot(0, np.cos(circle_u), np.sin(circle_u), 'k-', alpha=0.3)

            # Plot the state vector
            x, y, z = bloch_coordinates
            ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)

            # Add labels
            ax.text(1.3, 0, 0, '|+⟩', fontsize=12)
            ax.text(-1.3, 0, 0, '|-⟩', fontsize=12)
            ax.text(0, 1.3, 0, '|+i⟩', fontsize=12)
            ax.text(0, -1.3, 0, '|-i⟩', fontsize=12)
            ax.text(0, 0, 1.3, '|0⟩', fontsize=12)
            ax.text(0, 0, -1.3, '|1⟩', fontsize=12)

            # Set equal aspect ratio and remove axes
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
            ax.set_box_aspect([1,1,1])

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Set title
            plt.title(f"{title}\n{qubit_label}: ({x:.3f}, {y:.3f}, {z:.3f})",
                     fontsize=14, pad=20)

        except Exception as e:
            # Fallback to 2D plot if 3D fails
            plt.clf()
            ax = fig.add_subplot(111)

            # Draw 2D projection (x-z plane)
            circle = plt.Circle((0, 0), 1, fill=False, color='lightblue', linewidth=2)
            ax.add_patch(circle)

            # Draw axes
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

            # Plot state vector projection
            x, y, z = bloch_coordinates
            ax.arrow(0, 0, x, z, head_width=0.05, head_length=0.05,
                    fc='red', ec='red', linewidth=2)

            # Labels
            ax.text(1.1, 0, '|+⟩', fontsize=12, ha='center')
            ax.text(-1.1, 0, '|-⟩', fontsize=12, ha='center')
            ax.text(0, 1.1, '|0⟩', fontsize=12, ha='center')
            ax.text(0, -1.1, '|1⟩', fontsize=12, ha='center')

            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_aspect('equal')
            ax.set_title(f"{title} (X-Z projection)\n{qubit_label}: ({x:.3f}, {y:.3f}, {z:.3f})")

        plt.tight_layout()
        return fig

    def draw_circuit(self, circuit: QuantumCircuit, title: str = "Quantum Circuit") -> plt.Figure:
        """
        Create a circuit diagram visualization.

        Args:
            circuit: The quantum circuit to visualize
            title: Title for the plot

        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(12, 6), dpi=self.dpi)

        try:
            # Use Qiskit's circuit drawer
            circuit_img = circuit_drawer(circuit, output='mpl', style='iqp')
            if hasattr(circuit_img, 'figure'):
                plt.close(fig)  # Close our figure
                fig = circuit_img.figure
                fig.suptitle(title, fontsize=16, y=0.95)
            else:
                # Fallback: create a simple text representation
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, str(circuit), fontsize=10, ha='center', va='center',
                       transform=ax.transAxes, family='monospace')
                ax.set_title(title)
                ax.axis('off')
        except Exception as e:
            # Fallback to text representation
            ax = fig.add_subplot(111)
            circuit_str = str(circuit)
            ax.text(0.5, 0.5, circuit_str, fontsize=8, ha='center', va='center',
                   transform=ax.transAxes, family='monospace')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        return fig

    def create_state_comparison_plot(self, states_before: Dict[str, Any],
                                   states_after: Dict[str, Any],
                                   case_number: int) -> plt.Figure:
        """
        Create a comparison plot showing states before and after Hadamard gate.

        Args:
            states_before: Dictionary with q0 and q1 state info before Hadamard
            states_after: Dictionary with q0 and q1 state info after Hadamard
            case_number: Circuit case number

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi,
                                subplot_kw={'projection': '3d'})
        fig.suptitle(f'Quantum State Evolution - Case {case_number}', fontsize=16)

        # Plot q0 before Hadamard
        self._plot_bloch_on_axis(axes[0, 0], states_before['q0']['bloch_coordinates'],
                                'q0 Before Final Hadamard', states_before['q0']['state_string'])

        # Plot q0 after Hadamard
        self._plot_bloch_on_axis(axes[0, 1], states_after['q0']['bloch_coordinates'],
                                'q0 After Final Hadamard', states_after['q0']['state_string'])

        # Plot q1 before Hadamard
        self._plot_bloch_on_axis(axes[1, 0], states_before['q1']['bloch_coordinates'],
                                'q1 Before Final Hadamard', states_before['q1']['state_string'])

        # Plot q1 after Hadamard
        self._plot_bloch_on_axis(axes[1, 1], states_after['q1']['bloch_coordinates'],
                                'q1 After Final Hadamard', states_after['q1']['state_string'])

        plt.tight_layout()
        return fig

    def _plot_bloch_on_axis(self, ax, bloch_coordinates: Tuple[float, float, float],
                           title: str, state_string: str):
        """
        Plot a Bloch sphere on a given axis.

        Args:
            ax: Matplotlib 3D axis
            bloch_coordinates: (x, y, z) coordinates
            title: Title for the subplot
            state_string: String representation of the state
        """
        # Draw the Bloch sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')

        # Draw coordinate axes
        ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)

        # Plot the state vector
        x, y, z = bloch_coordinates
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=2)

        # Set equal aspect ratio and limits
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_box_aspect([1,1,1])

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set title
        ax.set_title(f"{title}\n{state_string}", fontsize=10)

    def create_measurement_results_plot(self, all_results: Dict[int, Dict[str, Any]]) -> plt.Figure:
        """
        Create a bar plot comparing measurement results across all cases.

        Args:
            all_results: Dictionary mapping case numbers to measurement results

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)

        cases = list(all_results.keys())
        prob_0 = [all_results[case]['probabilities'].get('0', 0) for case in cases]
        prob_1 = [all_results[case]['probabilities'].get('1', 0) for case in cases]

        x = np.arange(len(cases))
        width = 0.35

        # Probability plot
        bars1 = ax1.bar(x - width/2, prob_0, width, label='Measure 0', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, prob_1, width, label='Measure 1', alpha=0.8, color='red')

        ax1.set_xlabel('Circuit Case')
        ax1.set_ylabel('Probability')
        ax1.set_title('Measurement Probabilities by Circuit Case')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Case {case}' for case in cases])
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

        # Counts plot
        counts_0 = [all_results[case]['counts'].get('0', 0) for case in cases]
        counts_1 = [all_results[case]['counts'].get('1', 0) for case in cases]

        bars3 = ax2.bar(x - width/2, counts_0, width, label='Count 0', alpha=0.8, color='lightblue')
        bars4 = ax2.bar(x + width/2, counts_1, width, label='Count 1', alpha=0.8, color='lightcoral')

        ax2.set_xlabel('Circuit Case')
        ax2.set_ylabel('Measurement Counts')
        ax2.set_title('Measurement Counts by Circuit Case')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Case {case}' for case in cases])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_visualization(self, fig: plt.Figure, filename: str, directory: Path) -> str:
        """
        Save a matplotlib figure to the specified directory.

        Args:
            fig: Matplotlib figure to save
            filename: Name of the file (without extension)
            directory: Directory to save the file

        Returns:
            Full path to the saved file
        """
        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # Add .png extension if not present
        if not filename.endswith('.png'):
            filename += '.png'

        filepath = directory / filename

        # Save the figure
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        # Close the figure to free memory
        plt.close(fig)

        return str(filepath)

    def create_all_visualizations(self, case_number: int, circuit: QuantumCircuit,
                                analysis_results: Dict[str, Any],
                                save_directory: Path) -> Dict[str, str]:
        """
        Create and save all visualizations for a circuit case.

        Args:
            case_number: Circuit case number
            circuit: The quantum circuit
            analysis_results: Complete analysis results from StateAnalyzer
            save_directory: Directory to save images

        Returns:
            Dictionary mapping visualization types to file paths
        """
        saved_files = {}

        # Circuit diagram
        circuit_fig = self.draw_circuit(circuit, f"Deutsch Algorithm - Case {case_number}")
        circuit_path = self.save_visualization(circuit_fig, f"circuit_case_{case_number}", save_directory)
        saved_files['circuit'] = circuit_path

        # State comparison plot
        states_before = analysis_results['states_before_hadamard']
        states_after = analysis_results['states_after_hadamard']
        comparison_fig = self.create_state_comparison_plot(states_before, states_after, case_number)
        comparison_path = self.save_visualization(comparison_fig, f"states_case_{case_number}", save_directory)
        saved_files['state_comparison'] = comparison_path

        # Individual Bloch spheres
        bloch_files = []
        for qubit in ['q0', 'q1']:
            for timing in ['before', 'after']:
                if timing == 'before':
                    state_info = states_before[qubit]
                    title = f"Case {case_number} - {qubit.upper()} Before Final Hadamard"
                else:
                    state_info = states_after[qubit]
                    title = f"Case {case_number} - {qubit.upper()} After Final Hadamard"

                bloch_fig = self.create_bloch_sphere(
                    state_info['bloch_coordinates'],
                    title,
                    qubit
                )
                bloch_path = self.save_visualization(
                    bloch_fig,
                    f"bloch_{qubit}_{timing}_case_{case_number}",
                    save_directory
                )
                bloch_files.append(bloch_path)

        saved_files['bloch_spheres'] = bloch_files

        return saved_files