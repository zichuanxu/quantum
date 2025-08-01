# Quantum Computing

Implementation of Deutsch Algorithm and Quantum vs Classical SVM Comparison

## Overview

This project implements two main components for a quantum computing assignment:

1. **Deutsch Algorithm Implementation**: Analysis of four different quantum circuit cases demonstrating quantum parallelism
2. **SVM Comparison**: Performance comparison between Quantum Support Vector Machine (QSVM) and Classical Support Vector Machine (CSVM)

## Project Structure

```text
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
└── results/                    # Generated results (created automatically)
    ├── deutsch_results.md      # Deutsch algorithm report
    ├── svm_comparison.md       # SVM comparison report
    └── images/                 # Generated visualizations
        ├── deutsch/            # Deutsch algorithm images
        └── svm/                # SVM comparison images
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
python main.py --base-dir my_results  # Change output directory
```

## Output Files

After execution, the following files will be generated:

### Markdown Reports

- `results/deutsch_results.md`: Complete Deutsch algorithm analysis
- `results/svm_comparison.md`: Complete SVM comparison analysis

### Visualizations

- `results/images/deutsch/`: Bloch spheres, circuit diagrams, comparison plots
- `results/images/svm/`: Decision boundaries, performance charts, confusion matrices
