# Quantum Computing Assignment 3

Implementation of Deutsch Algorithm and Quantum vs Classical SVM Comparison

## Overview

This project implements two main components for a quantum computing assignment:

1. **Deutsch Algorithm Implementation**: Analysis of four different quantum circuit cases demonstrating quantum parallelism
2. **SVM Comparison**: Performance comparison between Quantum Support Vector Machine (QSVM) and Classical Support Vector Machine (CSVM)

## Project Structure

```
quantum/
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source code
│   ├── deutsch_algorithm/      # Deutsch algorithm implementation
│   │   ├── circuit_builder.py  # Quantum circuit construction
│   │   ├── state_analyzer.py   # Quantum state analysis
│   │   ├── visualizer.py       # Bloch sphere and circuit visualization
│   │   └── deutsch_workflow.py # Main execution workflow
│   ├── svm_comparison/         # SVM comparison implementation
│   │   ├── data_processor.py   # Data loading and preprocessing
│   │   ├── qsvm_classifier.py  # Quantum SVM implementation
│   │   ├── csvm_classifier.py  # Classical SVM implementation
│   │   ├── performance_analyzer.py # Performance analysis
│   │   ├── svm_visualizer.py   # SVM visualization
│   │   └── svm_workflow.py     # Main execution workflow
│   └── utils/                  # Utility modules
│       ├── file_manager.py     # File and directory management
│       └── markdown_generator.py # Report generation
├── tests/                      # Unit tests
├── results/                    # Generated results
│   ├── deutsch_results.md      # Deutsch algorithm report
│   ├── svm_comparison.md       # SVM comparison report
│   └── images/                 # Generated visualizations
│       ├── deutsch/            # Deutsch algorithm images
│       └── svm/                # SVM comparison images
```

## Installation

1. **Set up conda environment** (recommended):

   ```bash
   conda create -n quantum python=3.9
   conda activate quantum
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Complete Assignment Execution

Run the entire assignment (both Deutsch algorithm and SVM comparison):

```bash
python main.py
```

### Individual Components

Run only the Deutsch algorithm analysis:

```bash
python main.py --deutsch-only
```

Run only the SVM comparison:

```bash
python main.py --svm-only
```

### Customization Options

**Quantum Circuit Parameters:**

```bash
python main.py --shots 2048  # Use 2048 shots for quantum circuits
```

**SVM Comparison Parameters:**

```bash
# Compare different digit pairs
python main.py --digits 0 1 5 6  # Compare 0vs1 and 5vs6

# Test different kernels
python main.py --kernels linear rbf poly sigmoid

# Set random seed
python main.py --random-state 123
```

**Output Directory:**

```bash
python main.py --base-dir my_quantum_results
```

## Features

### Deutsch Algorithm Analysis

- **Four Circuit Cases**: Implementation of all four Deutsch algorithm variants
- **Quantum State Analysis**: Analysis of qubit states before and after Hadamard gates
- **Bloch Sphere Visualization**: 3D visualization of quantum states
- **Circuit Diagrams**: Visual representation of quantum circuits
- **Measurement Analysis**: Statistical analysis of measurement outcomes

### SVM Comparison

- **Data Processing**: Automatic loading and preprocessing of digits dataset
- **Dimensionality Reduction**: PCA reduction from 64 to 2 dimensions
- **QSVM Implementation**: Quantum SVM using quantum kernels
- **CSVM Implementation**: Classical SVM with multiple kernel options
- **Performance Analysis**: Comprehensive timing and accuracy comparison
- **Decision Boundaries**: Visualization of classification boundaries
- **Kernel Comparison**: Analysis of different kernel performances

### Visualization and Reporting

- **Automated Reporting**: Comprehensive markdown reports with embedded images
- **Professional Visualizations**: High-quality plots and charts
- **Organized Output**: Structured directory layout for easy navigation
- **Image Management**: Automatic image saving and referencing

## Output Files

After execution, the following files will be generated:

### Reports

- `results/deutsch_results.md`: Complete Deutsch algorithm analysis
- `results/svm_comparison.md`: Complete SVM comparison analysis

### Visualizations

- `results/images/deutsch/`: Bloch spheres, circuit diagrams, comparison plots
- `results/images/svm/`: Decision boundaries, performance charts, confusion matrices

## Technical Details

### Deutsch Algorithm Implementation

The implementation covers four cases:

1. **Case 1**: Identity function (constant 0)
2. **Case 2**: NOT function (balanced)
3. **Case 3**: Constant 1 function
4. **Case 4**: Complex oracle function

Each case includes:

- Quantum circuit construction
- State vector analysis before/after final Hadamard gate
- Bloch sphere coordinate calculation
- Measurement probability analysis

### SVM Comparison Implementation

The comparison includes:

- **Data**: sklearn digits dataset (digits 3vs4, 1vs2)
- **Preprocessing**: PCA dimensionality reduction to 2D
- **QSVM**: Quantum kernel-based classification
- **CSVM**: Classical SVM with linear, RBF, polynomial kernels
- **Metrics**: Accuracy, precision, recall, F1-score, training time

## Dependencies

Key dependencies include:

- `qiskit>=1.1.0`: Quantum computing framework
- `qiskit-aer>=0.14.2`: Quantum simulator
- `qiskit-machine-learning>=0.7.0`: Quantum machine learning
- `scikit-learn>=1.3.0`: Classical machine learning
- `matplotlib>=3.7.0`: Plotting and visualization
- `numpy>=1.24.0`: Numerical computing
- `seaborn>=0.12.0`: Statistical visualization

## Testing

Run unit tests:

```bash
python -m pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **Qiskit Installation**: If you encounter Qiskit installation issues, try:

   ```bash
   pip install --upgrade pip
   pip install qiskit qiskit-aer qiskit-machine-learning
   ```

2. **Memory Issues**: For large datasets or high-resolution visualizations:

   ```bash
   python main.py --shots 512  # Reduce quantum circuit shots
   ```

3. **QSVM Failures**: If QSVM training fails, the system automatically falls back to classical methods with quantum kernels.

### Performance Optimization

- Use fewer shots for faster quantum circuit execution (trade-off with accuracy)
- Reduce visualization resolution for faster image generation
- Use linear kernels for faster classical SVM training

## Assignment Requirements Compliance

This implementation fulfills all assignment requirements:

✅ **Deutsch Algorithm**: Four circuit cases with state analysis and visualization
✅ **SVM Comparison**: QSVM vs CSVM with timing and accuracy analysis
✅ **Datasets**: Uses sklearn digits dataset with specified digit pairs
✅ **Dimensionality Reduction**: PCA from 64 to 2 dimensions
✅ **Kernel Comparison**: Multiple kernel types tested
✅ **Visualization**: Decision boundaries and performance charts
✅ **Documentation**: Comprehensive markdown reports
✅ **No Terminal Output**: All results saved to files

## License

This project is created for educational purposes as part of a quantum computing assignment.

## Contact

For questions or issues related to this implementation, please refer to the generated reports and documentation.
