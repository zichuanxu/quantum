"""
Quantum state analyzer for Deutsch algorithm implementation.
Analyzes quantum states before and after Hadamard gates and extracts Bloch sphere coordinates.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import numpy as np
from typing import Dict, Tuple, List, Any
import math


class StateAnalyzer:
    """Analyzes quantum states during Deutsch algorithm execution."""

    def __init__(self):
        """Initialize StateAnalyzer."""
        self.backend = Aer.get_backend('statevector_simulator')

    def _create_circuit_before_final_hadamard(self, original_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Create a circuit that stops just before the final Hadamard gate on q0.

        Args:
            original_circuit: The complete Deutsch algorithm circuit

        Returns:
            Circuit without the final Hadamard and measurement
        """
        # Create new circuit with same dimensions
        qc = QuantumCircuit(original_circuit.num_qubits)

        # Copy all gates except the last Hadamard on q0 and measurement
        for instruction in original_circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits

            # Skip the final Hadamard on q0 and measurement operations
            # Get qubit index in a version-compatible way
            try:
                qubit_index = original_circuit.find_bit(qubits[0]).index
            except (AttributeError, TypeError):
                try:
                    qubit_index = qubits[0]._index
                except AttributeError:
                    qubit_index = 0  # Assume first qubit

            if (gate.name == 'h' and qubit_index == 0 and
                instruction == original_circuit.data[-2]):  # Second to last instruction
                break
            elif gate.name == 'measure':
                break
            else:
                qc.append(gate, qubits)

        return qc

    def _create_circuit_after_final_hadamard(self, original_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Create a circuit that includes the final Hadamard gate but no measurement.

        Args:
            original_circuit: The complete Deutsch algorithm circuit

        Returns:
            Circuit with final Hadamard but without measurement
        """
        # Create new circuit with same dimensions
        qc = QuantumCircuit(original_circuit.num_qubits)

        # Copy all gates except measurement
        for instruction in original_circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits

            # Skip measurement operations
            if gate.name == 'measure':
                break
            else:
                qc.append(gate, qubits)

        return qc

    def _get_statevector(self, circuit: QuantumCircuit) -> Statevector:
        """
        Get the statevector for a quantum circuit.

        Args:
            circuit: The quantum circuit to analyze

        Returns:
            Statevector of the circuit
        """
        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled)
        result = job.result()
        return result.get_statevector()

    def _extract_single_qubit_state(self, statevector: Statevector, qubit_index: int) -> DensityMatrix:
        """
        Extract the density matrix for a single qubit from the full statevector.

        Args:
            statevector: Full system statevector
            qubit_index: Index of the qubit to extract (0 or 1)

        Returns:
            Density matrix of the single qubit
        """
        # Convert statevector to density matrix
        full_dm = DensityMatrix(statevector)

        # Trace out the other qubit
        if qubit_index == 0:
            # Trace out qubit 1, keep qubit 0
            single_qubit_dm = partial_trace(full_dm, [1])
        else:
            # Trace out qubit 0, keep qubit 1
            single_qubit_dm = partial_trace(full_dm, [0])

        return single_qubit_dm

    def _density_matrix_to_bloch_coordinates(self, dm: DensityMatrix) -> Tuple[float, float, float]:
        """
        Convert a single-qubit density matrix to Bloch sphere coordinates.

        Args:
            dm: Single-qubit density matrix

        Returns:
            Tuple of (x, y, z) Bloch sphere coordinates
        """
        # Get the density matrix as a numpy array
        rho = dm.data

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Calculate Bloch coordinates using Tr(ρ * σ_i)
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))

        return (x, y, z)

    def _state_to_string(self, dm: DensityMatrix) -> str:
        """
        Convert a density matrix to a readable string representation.

        Args:
            dm: Single-qubit density matrix

        Returns:
            String representation of the quantum state
        """
        rho = dm.data

        # Check if it's a pure state (trace of ρ² ≈ 1)
        trace_rho_squared = np.real(np.trace(rho @ rho))

        if abs(trace_rho_squared - 1.0) < 1e-10:  # Pure state
            # Extract amplitudes for |0⟩ and |1⟩
            alpha = np.sqrt(rho[0, 0])
            beta_magnitude = np.sqrt(rho[1, 1])

            # Get phase from off-diagonal element
            if abs(rho[0, 1]) > 1e-10:
                phase = np.angle(rho[0, 1] / (alpha * beta_magnitude)) if beta_magnitude > 1e-10 else 0
                beta = beta_magnitude * np.exp(1j * phase)
            else:
                beta = beta_magnitude

            # Format the state string using ASCII characters
            if abs(alpha) < 1e-10:
                return "|1>"
            elif abs(beta) < 1e-10:
                return "|0>"
            elif abs(abs(alpha) - abs(beta)) < 1e-10:  # Equal superposition
                if abs(np.real(beta) - np.real(alpha)) < 1e-10:
                    return "|+> = (|0> + |1>)/sqrt(2)"
                else:
                    return "|-> = (|0> - |1>)/sqrt(2)"
            else:
                return f"{alpha:.3f}|0> + {beta:.3f}|1>"
        else:
            # Mixed state
            return f"Mixed state (purity: {trace_rho_squared:.3f})"

    def analyze_states_before_hadamard(self, circuit: QuantumCircuit) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze quantum states of q0 and q1 just before the final Hadamard gate.

        Args:
            circuit: The complete Deutsch algorithm circuit

        Returns:
            Tuple of (q0_state_info, q1_state_info) dictionaries
        """
        # Create circuit without final Hadamard
        circuit_before = self._create_circuit_before_final_hadamard(circuit)

        # Get statevector
        statevector = self._get_statevector(circuit_before)

        # Extract individual qubit states
        q0_dm = self._extract_single_qubit_state(statevector, 0)
        q1_dm = self._extract_single_qubit_state(statevector, 1)

        # Get Bloch coordinates
        q0_bloch = self._density_matrix_to_bloch_coordinates(q0_dm)
        q1_bloch = self._density_matrix_to_bloch_coordinates(q1_dm)

        # Create state info dictionaries
        q0_info = {
            'density_matrix': q0_dm,
            'bloch_coordinates': q0_bloch,
            'state_string': self._state_to_string(q0_dm),
            'statevector': statevector
        }

        q1_info = {
            'density_matrix': q1_dm,
            'bloch_coordinates': q1_bloch,
            'state_string': self._state_to_string(q1_dm),
            'statevector': statevector
        }

        return q0_info, q1_info

    def analyze_states_after_hadamard(self, circuit: QuantumCircuit) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze quantum states of q0 and q1 after the final Hadamard gate.

        Args:
            circuit: The complete Deutsch algorithm circuit

        Returns:
            Tuple of (q0_state_info, q1_state_info) dictionaries
        """
        # Create circuit with final Hadamard but no measurement
        circuit_after = self._create_circuit_after_final_hadamard(circuit)

        # Get statevector
        statevector = self._get_statevector(circuit_after)

        # Extract individual qubit states
        q0_dm = self._extract_single_qubit_state(statevector, 0)
        q1_dm = self._extract_single_qubit_state(statevector, 1)

        # Get Bloch coordinates
        q0_bloch = self._density_matrix_to_bloch_coordinates(q0_dm)
        q1_bloch = self._density_matrix_to_bloch_coordinates(q1_dm)

        # Create state info dictionaries
        q0_info = {
            'density_matrix': q0_dm,
            'bloch_coordinates': q0_bloch,
            'state_string': self._state_to_string(q0_dm),
            'statevector': statevector
        }

        q1_info = {
            'density_matrix': q1_dm,
            'bloch_coordinates': q1_bloch,
            'state_string': self._state_to_string(q1_dm),
            'statevector': statevector
        }

        return q0_info, q1_info

    def get_measurement_probabilities(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Calculate measurement probabilities for the final state.

        Args:
            circuit: The complete Deutsch algorithm circuit

        Returns:
            Dictionary of measurement outcome probabilities
        """
        # Create circuit without measurement
        circuit_no_measure = self._create_circuit_after_final_hadamard(circuit)

        # Get statevector
        statevector = self._get_statevector(circuit_no_measure)

        # Calculate probabilities for measuring q0
        # |00⟩ and |01⟩ states correspond to measuring 0 on q0
        # |10⟩ and |11⟩ states correspond to measuring 1 on q0
        prob_0 = abs(statevector[0])**2 + abs(statevector[1])**2
        prob_1 = abs(statevector[2])**2 + abs(statevector[3])**2

        return {
            '0': prob_0,
            '1': prob_1
        }

    def analyze_complete_evolution(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Perform complete state analysis for a Deutsch algorithm circuit.

        Args:
            circuit: The complete Deutsch algorithm circuit

        Returns:
            Dictionary containing all analysis results
        """
        # Analyze states before final Hadamard
        q0_before, q1_before = self.analyze_states_before_hadamard(circuit)

        # Analyze states after final Hadamard
        q0_after, q1_after = self.analyze_states_after_hadamard(circuit)

        # Get measurement probabilities
        measurement_probs = self.get_measurement_probabilities(circuit)

        return {
            'states_before_hadamard': {
                'q0': q0_before,
                'q1': q1_before
            },
            'states_after_hadamard': {
                'q0': q0_after,
                'q1': q1_after
            },
            'measurement_probabilities': measurement_probs
        }