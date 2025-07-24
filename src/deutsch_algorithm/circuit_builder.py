"""
Quantum circuit builder for Deutsch algorithm implementation.
Creates the four different circuit cases as specified in the assignment.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from typing import Dict, Any
import numpy as np


class CircuitBuilder:
    """Builds and executes quantum circuits for the Deutsch algorithm."""

    def __init__(self, shots: int = 1024):
        """Initialize CircuitBuilder with simulation parameters."""
        self.shots = shots
        self.backend = Aer.get_backend('aer_simulator')

    def create_circuit_case_1(self) -> QuantumCircuit:
        """
        Create Case 1: Identity function (constant 0 for both inputs).
        Oracle: No additional gates (identity operation).
        """
        # Create circuit with 2 qubits and 1 classical bit
        qc = QuantumCircuit(2, 1)

        # Initialize q1 to |1⟩ state
        qc.x(1)

        # Apply Hadamard gates to both qubits
        qc.h(0)
        qc.h(1)

        # Oracle for case 1: Identity (no gates needed)
        # The oracle does nothing, so f(0) = 0 and f(1) = 0

        # Apply final Hadamard gate to q0
        qc.h(0)

        # Measure q0
        qc.measure(0, 0)

        return qc

    def create_circuit_case_2(self) -> QuantumCircuit:
        """
        Create Case 2: NOT function (f(x) = NOT x).
        Oracle: CNOT gate with q0 as control and q1 as target.
        """
        # Create circuit with 2 qubits and 1 classical bit
        qc = QuantumCircuit(2, 1)

        # Initialize q1 to |1⟩ state
        qc.x(1)

        # Apply Hadamard gates to both qubits
        qc.h(0)
        qc.h(1)

        # Oracle for case 2: CNOT gate (NOT function)
        # This implements f(0) = 0, f(1) = 1
        qc.cx(0, 1)

        # Apply final Hadamard gate to q0
        qc.h(0)

        # Measure q0
        qc.measure(0, 0)

        return qc

    def create_circuit_case_3(self) -> QuantumCircuit:
        """
        Create Case 3: Constant function f(x) = 1.
        Oracle: X gate on q1, then CNOT, then X gate on q1 again.
        """
        # Create circuit with 2 qubits and 1 classical bit
        qc = QuantumCircuit(2, 1)

        # Initialize q1 to |1⟩ state
        qc.x(1)

        # Apply Hadamard gates to both qubits
        qc.h(0)
        qc.h(1)

        # Oracle for case 3: Constant function f(x) = 1
        # Apply X gate to q1, then CNOT, then X gate to q1 again
        qc.x(1)
        qc.cx(0, 1)
        qc.x(1)

        # Apply final Hadamard gate to q0
        qc.h(0)

        # Measure q0
        qc.measure(0, 0)

        return qc

    def create_circuit_case_4(self) -> QuantumCircuit:
        """
        Create Case 4: More complex oracle function.
        Oracle: CNOT, X gate on q1, then another CNOT.
        """
        # Create circuit with 2 qubits and 1 classical bit
        qc = QuantumCircuit(2, 1)

        # Initialize q1 to |1⟩ state
        qc.x(1)

        # Apply Hadamard gates to both qubits
        qc.h(0)
        qc.h(1)

        # Oracle for case 4: Complex oracle
        # CNOT, then X on q1, then CNOT again
        qc.cx(0, 1)
        qc.x(1)
        qc.cx(0, 1)

        # Apply final Hadamard gate to q0
        qc.h(0)

        # Measure q0
        qc.measure(0, 0)

        return qc

    def execute_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Execute a quantum circuit and return measurement results.

        Args:
            circuit: The quantum circuit to execute

        Returns:
            Dictionary containing measurement counts and probabilities
        """
        # Transpile circuit for the backend
        transpiled_circuit = transpile(circuit, self.backend)

        # Execute the circuit
        job = self.backend.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate probabilities
        total_shots = sum(counts.values())
        probabilities = {outcome: count/total_shots for outcome, count in counts.items()}

        return {
            'counts': counts,
            'probabilities': probabilities,
            'total_shots': total_shots,
            'circuit': circuit
        }

    def get_all_circuits(self) -> Dict[int, QuantumCircuit]:
        """
        Get all four Deutsch algorithm circuit cases.

        Returns:
            Dictionary mapping case numbers to quantum circuits
        """
        return {
            1: self.create_circuit_case_1(),
            2: self.create_circuit_case_2(),
            3: self.create_circuit_case_3(),
            4: self.create_circuit_case_4()
        }

    def execute_all_cases(self) -> Dict[int, Dict[str, Any]]:
        """
        Execute all four circuit cases and return results.

        Returns:
            Dictionary mapping case numbers to execution results
        """
        circuits = self.get_all_circuits()
        results = {}

        for case_num, circuit in circuits.items():
            results[case_num] = self.execute_circuit(circuit)

        return results

    def get_circuit_description(self, case_number: int) -> Dict[str, str]:
        """
        Get description of a specific circuit case.

        Args:
            case_number: The case number (1-4)

        Returns:
            Dictionary with circuit description details
        """
        descriptions = {
            1: {
                'name': 'Identity Function',
                'oracle_type': 'No oracle gates (identity)',
                'function': 'f(x) = 0 for all x',
                'expected_result': 'Always measure 0 (constant function)'
            },
            2: {
                'name': 'NOT Function',
                'oracle_type': 'CNOT gate',
                'function': 'f(x) = NOT x',
                'expected_result': 'Always measure 1 (balanced function)'
            },
            3: {
                'name': 'Constant 1 Function',
                'oracle_type': 'X-CNOT-X sequence',
                'function': 'f(x) = 1 for all x',
                'expected_result': 'Always measure 0 (constant function)'
            },
            4: {
                'name': 'Complex Oracle Function',
                'oracle_type': 'CNOT-X-CNOT sequence',
                'function': 'Complex oracle implementation',
                'expected_result': 'Measurement depends on oracle behavior'
            }
        }

        if case_number not in descriptions:
            raise ValueError(f"Invalid case number: {case_number}. Must be 1-4.")

        return descriptions[case_number]