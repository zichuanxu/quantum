# Quantum SVM vs Classical SVM Comparison

Generated on: 2025-07-26 08:17:26

---

## Overview

This report compares the performance of Quantum Support Vector Machine (QSVM) and Classical Support Vector Machine (CSVM) on digit classification tasks. The comparison includes execution time, accuracy, and decision boundary analysis.

## Experiment: Digits 3 vs 4 (linear kernel)

### Performance Comparison

| Metric | QSVM | CSVM | Difference |
| --- | --- | --- | --- |
| Training Time (s) | 21.0757 | 0.0008 | 21.0749 |
| Accuracy (%) | 0.54 | 1.00 | -0.46 |

### Decision Boundary Visualization

![Decision Boundaries](images/svm/boundaries_3vs4_linear.png)

*Decision boundaries for digits 3 vs 4*

### Analysis

QSVM took 26434.8x longer to train than CSVM.

CSVM achieved 0.46% higher accuracy than QSVM.

## Experiment: Digits 3 vs 4 (rbf kernel)

### Performance Comparison

| Metric | QSVM | CSVM | Difference |
| --- | --- | --- | --- |
| Training Time (s) | 24.5719 | 0.0010 | 24.5709 |
| Accuracy (%) | 0.54 | 1.00 | -0.46 |

### Decision Boundary Visualization

![Decision Boundaries](images/svm/boundaries_3vs4_rbf.png)

*Decision boundaries for digits 3 vs 4*

### Analysis

QSVM took 24721.0x longer to train than CSVM.

CSVM achieved 0.46% higher accuracy than QSVM.

## Experiment: Digits 1 vs 2 (linear kernel)

### Performance Comparison

| Metric | QSVM | CSVM | Difference |
| --- | --- | --- | --- |
| Training Time (s) | 24.1361 | 0.0034 | 24.1328 |
| Accuracy (%) | 0.51 | 0.87 | -0.36 |

### Decision Boundary Visualization

![Decision Boundaries](images/svm/boundaries_1vs2_linear.png)

*Decision boundaries for digits 1 vs 2*

### Analysis

QSVM took 7194.0x longer to train than CSVM.

CSVM achieved 0.36% higher accuracy than QSVM.

## Experiment: Digits 1 vs 2 (rbf kernel)

### Performance Comparison

| Metric | QSVM | CSVM | Difference |
| --- | --- | --- | --- |
| Training Time (s) | 26.5439 | 0.0022 | 26.5417 |
| Accuracy (%) | 0.51 | 0.87 | -0.36 |

### Decision Boundary Visualization

![Decision Boundaries](images/svm/boundaries_1vs2_rbf.png)

*Decision boundaries for digits 1 vs 2*

### Analysis

QSVM took 12004.9x longer to train than CSVM.

CSVM achieved 0.36% higher accuracy than QSVM.

## Kernel Comparison Analysis

### Kernels for Digits 3 vs 4

| Kernel | Accuracy | Training Time (s) | F1-Score |
| --- | --- | --- | --- |
| Linear | 1.000 | 0.0008 | 1.000 |
| Rbf | 1.000 | 0.0009 | 1.000 |
| Poly | 0.991 | 0.0011 | 0.991 |

![Kernel Comparison for Digits 3 vs 4](images/svm/kernels_3vs4.png)

*Performance comparison across different kernel types*

### Kernels for Digits 1 vs 2

| Kernel | Accuracy | Training Time (s) | F1-Score |
| --- | --- | --- | --- |
| Linear | 0.870 | 0.0034 | 0.877 |
| Rbf | 0.870 | 0.0024 | 0.879 |
| Poly | 0.889 | 0.0019 | 0.898 |

![Kernel Comparison for Digits 1 vs 2](images/svm/kernels_1vs2.png)

*Performance comparison across different kernel types*

## Summary Analysis

**Overall Performance Summary:**

- Average QSVM training time: 24.0819s

- Average CSVM training time: 0.0018s

- Average QSVM accuracy: 0.52%

- Average CSVM accuracy: 0.94%

**Key Findings:**

- QSVM generally requires more training time than CSVM

- CSVM maintains competitive or superior accuracy performance

## Key Findings and Conclusions

- Total experiments conducted: 4
- QSVM accuracy wins: 0/4 experiments
- Average QSVM accuracy: 0.523
- Average CSVM accuracy: 0.935
- Average QSVM training time: 24.0819s
- Average CSVM training time: 0.0018s
