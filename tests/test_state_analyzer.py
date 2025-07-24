"""
Unit tests for StateAnalyzer class.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from quantum.src.deutsch_algorithm.state_analyzer import StateAnalyzer
from quantum.src.deutsch_algorithm.circuit_builder import CircuitBuilder


class TestStateAnalyzer(unittest.TestCase):
    """Test cases for StateAnalyzer functionality."""

    def setUp(self):
        """Set up test environment."""
        self.analyzer = StateAnalyzer()
        self.builder = CircuitBuilder()

    def test_bloch_coordinates_calculation(self):
        """Test Bloch sphere coordinate calculation."""
        # Test |0⟩ state
        dm_0 = DensityMatrix.from_label('0')
        x, y, z = self.analyzer._density_matrix_to_bloch_coordinates(dm_0)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 0, places=10)
        self.assertAlmostEqual(z, 1, places=10)

        # Test |1⟩ state
        dm_1 = DensityMatrix.from_label('1')
        x, y, z = self.analyzer._density_matrix_to_bloch_coordinates(dm_1)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 0, places=10)
        self.assertAlmostEqual(z, -1, places=10)

        # Test |+⟩ state
        dm_plus = DensityMatrix.from_label('+')
        x, y, z = self.analyzer._density_matrix_to_bloch_coordinates(dm_plus)
        self.assertAlmostEqual(x, 1, places=10)
        self.assertAlmostEqual(y, 0, places=10)
        self.assertAlmostEqual(z, 0, places=10)

    def test_state_string_representation(self):
        """Test quantum state string representation."""
        # Test |0⟩ state
        dm_0 = DensityMatrix.from_label('0')
        state_str = self.analyzer._state_to_string(dm_0)
        self.assertEqual(state_str, "|0⟩")

        # Test |1⟩ state
        dm_1 = DensityMatrix.from_label('1')
        state_str = self.analyzer._state_to_string(dm_1)
        self.assertEqual(state_str, "|1⟩")

        # Test |+⟩ state
        dm_plus = DensityMatrix.from_label('+')
        state_str = self.analyzer._state_to_string(dm_plus)
        self.assertIn("|+⟩", state_str)

    def test_measurement_probabilities(self):
        """Test measurement probability calculation."""
        # Test with a simple circuit
        circuit = self.builder.create_circuit_case_1()
        probs = self.analyzer.get_measurement_probabilities(circuit)

        # Check structure
        self.assertIn('0', probs)
        self.assertIn('1', probs)

        # Check probabilities sum to 1
        total_prob = probs['0'] + probs['1']
        self.assertAlmostEqual(total_prob, 1.0, places=10)

        # Check probabilities are non-negative
        self.assertGreaterEqual(probs['0'], 0)
        self.assertGreaterEqual(probs['1'], 0)

    def test_state_analysis_before_hadamard(self):
        """Test state analysis before final Hadamard gate."""
        circuit = self.builder.create_circuit_case_1()
        q0_info, q1_info = self.analyzer.analyze_states_before_hadamard(circuit)

        # Check structure of returned information
        for info in [q0_info, q1_info]:
            self.assertIn('density_matrix', info)
            self.assertIn('bloch_coordinates', info)
            self.assertIn('state_string', info)
            self.assertIn('statevector', info)

        # Check Bloch coordinates are tuples of 3 floats
        self.assertEqual(len(q0_info['bloch_coordinates']), 3)
        self.assertEqual(len(q1_info['bloch_coordinates']), 3)

        # Check that coordinates are within valid range [-1, 1]
        for coord in q0_info['bloch_coordinates'] + q1_info['bloch_coordinates']:
            self.assertGreaterEqual(coord, -1.1)  # Small tolerance for numerical errors
            self.assertLessEqual(coord, 1.1)

    def test_state_analysis_after_hadamard(self):
        """Test state analysis after final Hadamard gate."""
        circuit = self.builder.create_circuit_case_1()
        q0_info, q1_info = self.analyzer.analyze_states_after_hadamard(circuit)

        # Check structure of returned information
        for info in [q0_info, q1_info]:
            self.assertIn('density_matrix', info)
            self.assertIn('bloch_coordinates', info)
            self.assertIn('state_string', info)
            self.assertIn('statevector', info)

        # Check Bloch coordinates are valid
        self.assertEqual(len(q0_info['bloch_coordinates']), 3)
        self.assertEqual(len(q1_info['bloch_coordinates']), 3)

    def test_complete_evolution_analysis(self):
        """Test complete evolution analysis."""
        circuit = self.builder.create_circuit_case_1()
        analysis = self.analyzer.analyze_complete_evolution(circuit)

        # Check top-level structure
        self.assertIn('states_before_hadamard', analysis)
        self.assertIn('states_after_hadamard', analysis)
        self.assertIn('measurement_probabilities', analysis)

        # Check states structure
        before_states = analysis['states_before_hadamard']
        after_states = analysis['states_after_hadamard']

        self.assertIn('q0', before_states)
        self.assertIn('q1', before_states)
        self.assertIn('q0', after_states)
        self.assertIn('q1', after_states)

        # Check measurement probabilities
        probs = analysis['measurement_probabilities']
        self.assertIn('0', probs)
        self.assertIn('1', probs)
        self.assertAlmostEqual(probs['0'] + probs['1'], 1.0, places=10)

    def test_different_circuits_produce_different_states(self):
        """Test that different circuit cases produce different state analyses."""
        circuit1 = self.builder.create_circuit_case_1()
        circuit2 = self.builder.create_circuit_case_2()

        analysis1 = self.analyzer.analyze_complete_evolution(circuit1)
        analysis2 = self.analyzer.analyze_complete_evolution(circuit2)

        # The measurement probabilities should be different for different cases
        probs1 = analysis1['measurement_probabilities']
        probs2 = analysis2['measurement_probabilities']

        # At least one probability should be significantly different
        prob_diff = abs(probs1['0'] - probs2['0'])
        self.assertGreater(prob_diff, 0.1)  # Should be substantially different

    def test_circuit_modification_methods(self):
        """Test internal circuit modification methods."""
        original_circuit = self.builder.create_circuit_case_1()

        # Test circuit before final Hadamard
        circuit_before = self.analyzer._create_circuit_before_final_hadamard(original_circuit)
        self.assertIsInstance(circuit_before, QuantumCircuit)
        self.assertEqual(circuit_before.num_qubits, 2)

        # Test circuit after final Hadamard
        circuit_after = self.analyzer._create_circuit_after_final_hadamard(original_circuit)
        self.assertIsInstance(circuit_after, QuantumCircuit)
        self.assertEqual(circuit_after.num_qubits, 2)

        # The "after" circuit should have more gates than "before" circuit
        self.assertGreaterEqual(len(circuit_after.data), len(circuit_before.data))


if __name__ == '__main__':
    unittest.main()