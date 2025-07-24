"""
Unit tests for CircuitBuilder class.
"""

import unittest
from qiskit import QuantumCircuit
from quantum.src.deutsch_algorithm.circuit_builder import CircuitBuilder


class TestCircuitBuilder(unittest.TestCase):
    """Test cases for CircuitBuilder functionality."""

    def setUp(self):
        """Set up test environment."""
        self.builder = CircuitBuilder(shots=100)  # Use fewer shots for testing

    def test_circuit_creation(self):
        """Test that all circuit cases are created correctly."""
        # Test each circuit case
        circuit1 = self.builder.create_circuit_case_1()
        circuit2 = self.builder.create_circuit_case_2()
        circuit3 = self.builder.create_circuit_case_3()
        circuit4 = self.builder.create_circuit_case_4()

        # Check that all circuits are QuantumCircuit instances
        self.assertIsInstance(circuit1, QuantumCircuit)
        self.assertIsInstance(circuit2, QuantumCircuit)
        self.assertIsInstance(circuit3, QuantumCircuit)
        self.assertIsInstance(circuit4, QuantumCircuit)

        # Check circuit dimensions (2 qubits, 1 classical bit)
        for circuit in [circuit1, circuit2, circuit3, circuit4]:
            self.assertEqual(circuit.num_qubits, 2)
            self.assertEqual(circuit.num_clbits, 1)

    def test_circuit_execution(self):
        """Test circuit execution returns proper results."""
        circuit = self.builder.create_circuit_case_1()
        result = self.builder.execute_circuit(circuit)

        # Check result structure
        self.assertIn('counts', result)
        self.assertIn('probabilities', result)
        self.assertIn('total_shots', result)
        self.assertIn('circuit', result)

        # Check that total shots matches expected
        self.assertEqual(result['total_shots'], 100)

        # Check that probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=10)

    def test_get_all_circuits(self):
        """Test getting all circuits at once."""
        circuits = self.builder.get_all_circuits()

        # Check that we get all 4 circuits
        self.assertEqual(len(circuits), 4)
        self.assertIn(1, circuits)
        self.assertIn(2, circuits)
        self.assertIn(3, circuits)
        self.assertIn(4, circuits)

        # Check that all are QuantumCircuit instances
        for circuit in circuits.values():
            self.assertIsInstance(circuit, QuantumCircuit)

    def test_execute_all_cases(self):
        """Test executing all circuit cases."""
        results = self.builder.execute_all_cases()

        # Check that we get results for all 4 cases
        self.assertEqual(len(results), 4)

        # Check structure of each result
        for case_num in [1, 2, 3, 4]:
            self.assertIn(case_num, results)
            result = results[case_num]
            self.assertIn('counts', result)
            self.assertIn('probabilities', result)
            self.assertIn('total_shots', result)

    def test_circuit_descriptions(self):
        """Test circuit description retrieval."""
        # Test valid case numbers
        for case_num in [1, 2, 3, 4]:
            desc = self.builder.get_circuit_description(case_num)
            self.assertIn('name', desc)
            self.assertIn('oracle_type', desc)
            self.assertIn('function', desc)
            self.assertIn('expected_result', desc)

        # Test invalid case number
        with self.assertRaises(ValueError):
            self.builder.get_circuit_description(5)

    def test_circuit_case_differences(self):
        """Test that different cases produce different circuits."""
        circuits = self.builder.get_all_circuits()

        # Convert circuits to strings for comparison
        circuit_strings = {}
        for case_num, circuit in circuits.items():
            circuit_strings[case_num] = str(circuit)

        # Check that circuits are different
        self.assertNotEqual(circuit_strings[1], circuit_strings[2])
        self.assertNotEqual(circuit_strings[1], circuit_strings[3])
        self.assertNotEqual(circuit_strings[1], circuit_strings[4])
        self.assertNotEqual(circuit_strings[2], circuit_strings[3])
        self.assertNotEqual(circuit_strings[2], circuit_strings[4])
        self.assertNotEqual(circuit_strings[3], circuit_strings[4])


if __name__ == '__main__':
    unittest.main()