# Deutsch Algorithm Implementation Results

Generated on: 2025-08-01 09:26:59

---

## Overview

This report presents the implementation and analysis of the Deutsch algorithm across four different quantum circuit configurations. Each case demonstrates different oracle functions and their impact on quantum state evolution.

## Case 1: Constant Zero Function

### Circuit Description

Oracle Type: No oracle gates (identity)

![Circuit Case 1](images/deutsch/circuit_case_1.png)

*Quantum circuit for Case 1*

### Quantum State Analysis

**States before final Hadamard gate:**

- q0: |+> = (|0> + |1>)/sqrt(2)

- q1: |-> = (|0> - |1>)/sqrt(2)

**States after final Hadamard gate:**

- q0: |0>

- q1: |-> = (|0> - |1>)/sqrt(2)

### Bloch Sphere Visualizations

![Bloch Sphere 1](images/deutsch/bloch_q0_before_case_1.png)

*Bloch sphere representation for qubit 0*

![Bloch Sphere 2](images/deutsch/bloch_q0_after_case_1.png)

*Bloch sphere representation for qubit 1*

![Bloch Sphere 3](images/deutsch/bloch_q1_before_case_1.png)

*Bloch sphere representation for qubit 2*

![Bloch Sphere 4](images/deutsch/bloch_q1_after_case_1.png)

*Bloch sphere representation for qubit 3*

### Measurement Results

- Outcome '0': 1024/1024 (100.0%)

## Case 2: NOT Function (Balanced)

### Circuit Description

Oracle Type: CNOT gate

![Circuit Case 2](images/deutsch/circuit_case_2.png)

*Quantum circuit for Case 2*

### Quantum State Analysis

**States before final Hadamard gate:**

- q0: |-> = (|0> - |1>)/sqrt(2)

- q1: |-> = (|0> - |1>)/sqrt(2)

**States after final Hadamard gate:**

- q0: |1>

- q1: |-> = (|0> - |1>)/sqrt(2)

### Bloch Sphere Visualizations

![Bloch Sphere 1](images/deutsch/bloch_q0_before_case_2.png)

*Bloch sphere representation for qubit 0*

![Bloch Sphere 2](images/deutsch/bloch_q0_after_case_2.png)

*Bloch sphere representation for qubit 1*

![Bloch Sphere 3](images/deutsch/bloch_q1_before_case_2.png)

*Bloch sphere representation for qubit 2*

![Bloch Sphere 4](images/deutsch/bloch_q1_after_case_2.png)

*Bloch sphere representation for qubit 3*

### Measurement Results

- Outcome '1': 1024/1024 (100.0%)

## Case 3: Constant One Function

### Circuit Description

Oracle Type: X gate on ancilla

![Circuit Case 3](images/deutsch/circuit_case_3.png)

*Quantum circuit for Case 3*

### Quantum State Analysis

**States before final Hadamard gate:**

- q0: |+> = (|0> + |1>)/sqrt(2)

- q1: |-> = (|0> - |1>)/sqrt(2)

**States after final Hadamard gate:**

- q0: |0>

- q1: |-> = (|0> - |1>)/sqrt(2)

### Bloch Sphere Visualizations

![Bloch Sphere 1](images/deutsch/bloch_q0_before_case_3.png)

*Bloch sphere representation for qubit 0*

![Bloch Sphere 2](images/deutsch/bloch_q0_after_case_3.png)

*Bloch sphere representation for qubit 1*

![Bloch Sphere 3](images/deutsch/bloch_q1_before_case_3.png)

*Bloch sphere representation for qubit 2*

![Bloch Sphere 4](images/deutsch/bloch_q1_after_case_3.png)

*Bloch sphere representation for qubit 3*

### Measurement Results

- Outcome '0': 1024/1024 (100.0%)

## Case 4: Identity Function (Balanced)

### Circuit Description

Oracle Type: X-CNOT-X sequence

![Circuit Case 4](images/deutsch/circuit_case_4.png)

*Quantum circuit for Case 4*

### Quantum State Analysis

**States before final Hadamard gate:**

- q0: |-> = (|0> - |1>)/sqrt(2)

- q1: |-> = (|0> - |1>)/sqrt(2)

**States after final Hadamard gate:**

- q0: |1>

- q1: |-> = (|0> - |1>)/sqrt(2)

### Bloch Sphere Visualizations

![Bloch Sphere 1](images/deutsch/bloch_q0_before_case_4.png)

*Bloch sphere representation for qubit 0*

![Bloch Sphere 2](images/deutsch/bloch_q0_after_case_4.png)

*Bloch sphere representation for qubit 1*

![Bloch Sphere 3](images/deutsch/bloch_q1_before_case_4.png)

*Bloch sphere representation for qubit 2*

![Bloch Sphere 4](images/deutsch/bloch_q1_after_case_4.png)

*Bloch sphere representation for qubit 3*

### Measurement Results

- Outcome '1': 1024/1024 (100.0%)

## Cross-Case Comparison

The following visualization compares measurement results across all circuit cases:

![Measurement Results Comparison](images/deutsch/measurement_comparison.png)

*Comparison of measurement probabilities and counts across all four cases*

## Theoretical Analysis

The Deutsch algorithm demonstrates quantum parallelism by determining whether a function is constant or balanced with just one quantum query, compared to two classical queries required in the worst case.

**Key Observations:**

- Initial state after first Hadamard gates: All cases start with |+>|-> state
- Cases 1 and 3 implement constant functions (f(x) = 0 and f(x) = 1 respectively)
- Cases 2 and 4 implement balanced functions (f(x) = NOT x and f(x) = x respectively)
- Phase kickback mechanism: Balanced functions flip q0 from |+> to |-> before final Hadamard
- Constant functions: q0 remains in |+> state, final measurement gives 0
- Balanced functions: q0 flips to |-> state, final measurement gives 1
- The final measurement of q0 reveals the function type: 0 for constant, 1 for balanced
- Quantum superposition allows simultaneous evaluation of the function on all inputs

### Phase Kickback Mechanism

The Deutsch algorithm relies on the quantum phase kickback effect to distinguish between constant and balanced functions:

- All circuits start with q0 and q1 in superposition states |+> and |-> respectively
- Constant functions (Cases 1 & 3): Oracle doesn't change q0's state, remains |+>
- Balanced functions (Cases 2 & 4): Oracle causes phase kickback, flipping q0 to |->
- Final Hadamard on q0: |+> → |0> (measure 0), |-> → |1> (measure 1)
- This allows determining function type with just one quantum query
