# KITE: Kernel-based Inference of Trait Epistasis

KITE is a Python package for detecting epistatic interactions in quantitative trait loci (QTL) data using kernel-based methods. The package provides a user-friendly framework for identifying non-linear interactions between genetic variants that influence complex traits.

## Proposal

You can view the detailed proposal for the KITE project [here](https://repo.fufoundation.co/Research/KITE.pdf).

## Features

- Kernel-based approach for capturing non-linear interactions between genotypes
- Dimensionality reduction using Kernel Principal Component Analysis (KPCA)
- Epistasis detection using Kernel Ridge Regression
- Neural network-based approach for epistasis detection (EpistasisDetector)
- Preprocessing utilities for handling missing data and encoding genotypes
- Command-line interface for easy integration into existing workflows
- Modular and extensible design for incorporating new methods and algorithms

## Installation

You can install the `biokite` package using pip:

```
pip install biokite
```

## Usage

### Command-line interface

The `biokite` package provides a command-line interface for running the epistasis detection algorithms. To use the CLI, run the following command:

```
biokite --data <path_to_data> --output <path_to_output> [--hidden-dims <hidden_dimensions>] [--dropout-rate <dropout_rate>] [--learning-rate <learning_rate>] [--batch-size <batch_size>] [--num-epochs <num_epochs>] [--n-components <n_components>] [--kernel <kernel>] [--alpha <alpha>] [--gamma <gamma>]
```

Arguments:
- `--data`: Path to the input data file (required).
- `--output`: Path to save the trained model (required).
- `--hidden-dims`: Hidden layer dimensions for the EpistasisDetector (default: [64, 32]).
- `--dropout-rate`: Dropout rate for the EpistasisDetector (default: 0.5).
- `--learning-rate`: Learning rate for the EpistasisDetector (default: 0.001).
- `--batch-size`: Batch size for the EpistasisDetector (default: 32).
- `--num-epochs`: Number of epochs for the EpistasisDetector (default: 50).
- `--n-components`: Number of principal components for KPCA (default: 10).
- `--kernel`: Kernel function for KPCA and Kernel Ridge Regression (default: 'rbf').
- `--alpha`: Regularization parameter for Kernel Ridge Regression (default: 1.0).
- `--gamma`: Kernel coefficient for the RBF kernel (default: None).

### Python API

You can also use the `biokite` package directly in your Python scripts or Jupyter notebooks. Here's an example of how to use the KITE detector:

```python
from biokite.detector import KITEDetector, preprocess_genotype_data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic genotype-phenotype data
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=0, n_repeated=0, n_classes=2, random_state=42)

# Preprocess the genotype data
X = preprocess_genotype_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KITE detector
kite_detector = KITEDetector(n_components=10, kernel='rbf', alpha=1.0, gamma=None)

# Train the KITE detector
kite_detector.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = kite_detector.predict(X_test)

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
```

## Data Format

The input data should be in CSV format, where each row represents a sample and the columns contain the genotype features and the phenotype label. The genotype features can be either numerical or categorical, and the phenotype label should be a continuous value.

Example:
```
SNP1,SNP2,SNP3,...,SNPn,Phenotype
0,1,2,1,0,1.5
1,0,1,2,1,2.3
2,1,0,0,1,0.8
...
```

## Terminology

- QTL: Quantitative Trait Locus, a region of the genome that is associated with a particular quantitative trait.
- Epistasis: The interaction between two or more genes that affects a phenotype.
- Kernel: A function that measures the similarity between pairs of data points in a high-dimensional space.
- KPCA: Kernel Principal Component Analysis, a non-linear dimensionality reduction technique.
- KRR: Kernel Ridge Regression, a regression method that uses a kernel function to capture non-linear relationships.

## API Design

The `biokite` package provides a high-level API for detecting epistatic interactions in QTL data. The main components of the API are:

- `KITEDetector`: A class that implements the kernel-based approach for epistasis detection using KPCA and KRR.
  - `__init__(n_components, kernel, alpha, gamma)`: Initializes the KITEDetector with the specified hyperparameters.
  - `fit(X, y)`: Fits the KITEDetector to the input data X and labels y.
  - `predict(X)`: Predicts the output for the input data X using the trained KITEDetector.
  - `score(X, y)`: Computes the R-squared score for the input data X and labels y.

- `EpistasisDetector`: A class that implements a neural network-based approach for epistasis detection.
  - `__init__(input_dim, hidden_dims, output_dim, dropout_rate)`: Initializes the EpistasisDetector with the specified architecture.
  - `forward(x)`: Performs a forward pass through the neural network.

- `run_epistasis_detector(X, y, hidden_dims, dropout_rate, learning_rate, batch_size, num_epochs, device)`: Trains and evaluates the EpistasisDetector on the input data X and labels y.

- `preprocess_genotype_data(X, encoding)`: Preprocesses the genotype data X using the specified encoding scheme.

- `select_best_kernel(X, y, kernels, param_grid)`: Selects the best kernel function and hyperparameters using grid search and cross-validation.

- `visualize_top_interactions(X, y, kite_detector, top_k)`: Visualizes the top k epistatic interactions identified by the KITEDetector.

## Roadmap

The biokite package currently supports the following features:
- [x] Kernel-based epistasis detection using Kernel PCA and Kernel Ridge Regression
- [x] Neural network-based epistasis detection using the EpistasisDetector class
- [x] Preprocessing of genotype data, including missing value imputation and one-hot encoding
- [x] Command-line interface for running epistasis detection algorithms

We are actively working on adding the following features:
- [ ] Cross-validation and model selection techniques for kernel selection and hyperparameter tuning
- [ ] Support for additional genotype encoding schemes, such as orthogonal polynomial coding and haplotype-based encodings
- [ ] Implementation of string kernels and graph kernels to capture complex genotype relationships
- [ ] Incorporation of prior biological knowledge into the kernel design
- [ ] Additional dimensionality reduction techniques, such as kernel CCA and kernel ICA
- [ ] Visualization tools for interpreting the results of epistasis detection
- [ ] Integration with popular bioinformatics libraries, such as BioPython and Scanpy
  
  Here's an updated Roadmap section with more Rust-related features and ideas:

## Roadmap

The biokite package currently supports the following features:
- [x] Kernel-based epistasis detection using Kernel PCA and Kernel Ridge Regression
- [x] Neural network-based epistasis detection using the EpistasisDetector class
- [x] Preprocessing of genotype data, including missing value imputation and one-hot encoding
- [x] Command-line interface for running epistasis detection algorithms

We are actively working on adding the following features:
- [ ] Cross-validation and model selection techniques for kernel selection and hyperparameter tuning
- [ ] Support for additional genotype encoding schemes, such as orthogonal polynomial coding and haplotype-based encodings
- [ ] Implementation of string kernels and graph kernels to capture complex genotype relationships
- [ ] Incorporation of prior biological knowledge into the kernel design
- [ ] Additional dimensionality reduction techniques, such as kernel CCA and kernel ICA
- [ ] Visualization tools for interpreting the results of epistasis detection
- [ ] Integration with popular bioinformatics libraries, such as BioPython and Scanpy

We are also planning to use Rust for performance-critical components and computationally intensive tasks in the future. The Python version of biokite will continue to be maintained and updated in parallel with our Rust development efforts.

Rust-related todos:
- [ ] Implement the core algorithms (Kernel PCA, Kernel Ridge Regression) in Rust
- [ ] Develop Rust-based utilities for data preprocessing and feature encoding
- [ ] Create Rust bindings for the Python package to enable seamless integration
- [ ] Benchmark the performance of the Rust implementation against the Python version!
- [ ] Explore the use of Rust's parallel and concurrent programming features to further optimize performance
- [ ] Implement Rust-based file parsers for common bioinformatics file formats (e.g., FASTA, FASTQ, VCF)
- [ ] Develop a Rust library for efficient computation of various genetic distance measures
- [ ] Investigate the use of Rust's SIMD (Single Instruction Multiple Data) capabilities for accelerating numerical computations
- [ ] Create a Rust-based framework for distributed computing in bioinformatics, leveraging Rust's strong concurrency primitives
- [ ] Explore the integration of Rust with big data processing frameworks like Apache Spark or Apache Arrow for handling large-scale genomic datasets
- [ ] Develop a Rust-based server for hosting RESTful APIs for bioinformatics tools and services
- [ ] Investigate the use of Rust's WebAssembly (WASM) support for building interactive web-based visualizations of epistatic interactions and QTL analysis results

Please stay tuned for updates and new releases of the `biokite` package!

## Contributing

Contributions to the `biokite` package are welcome! If you find a bug, have a feature request, or want to contribute code, please open an issue or submit a pull request on the GitHub repository.

## License

The `biokite` package is released under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions, suggestions, or collaborations, please contact the package maintainer:

- Xiang Fu
- xfu@bu.edu
- Boston University Faculty of Computing and Data Sciences