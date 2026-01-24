# AI-MNIST-2-3

A Java implementation of a Multi-Layer Perceptron (MLP) neural network for binary classification of MNIST handwritten digits 2 and 3.

## Overview

This project implements a fully connected feedforward neural network with backpropagation training for distinguishing between handwritten digits 2 and 3 from the MNIST dataset. The implementation includes data augmentation techniques, configurable network architectures, and threshold optimization for binary classification.

## Features

- **Custom MLP Implementation**: Fully connected neural network with configurable hidden layers
- **Multiple Activation Functions**: 
  - Sigmoid
  - ReLU (Rectified Linear Unit)
  - Leaky ReLU
  - Step function
- **Data Augmentation**: Image transformations including rotation, elastic deformation, and Gaussian noise
- **Training Features**:
  - Backpropagation with gradient descent
  - Early stopping with patience parameter
  - Optimal threshold optimization for binary classification
  - Model serialization/deserialization
- **Matrix Operations**: Custom matrix library for neural network computations

## Project Structure

```
AI-MNIST-2-3/
├── src/
│   ├── P4.java                           # Main prediction program
│   ├── apps/                             # Training applications
│   ├── math/                             # Matrix and array utilities
│   │   ├── Matrix.java
│   │   └── Array.java
│   ├── neural/                           # Neural network core
│   │   ├── MLP.java                      # Multi-Layer Perceptron implementation
│   │   └── activation/                   # Activation functions
│   │       ├── IDifferentiableFunction.java
│   │       ├── Sigmoid.java
│   │       ├── ReLU.java
│   │       ├── LeakyReLU.java
│   │       └── Step.java
│   ├── ml/                               # Machine learning utilities
│   │   ├── data/                         # Data processing
│   │   │   ├── DataSetBuilder.java
│   │   │   └── ImageAugmentation.java
│   │   └── training/                     # Training utilities
│   │       ├── config/
│   │       ├── result/
│   │       └── threshold/
│   └── utils/                            # Utility classes
│       ├── CSVReader.java
│       └── RandomProvider.java
├── data/                                 # Dataset files
├── bin/                                  # Compiled classes
├── mse_results/                          # Training results
├── report/                               # Documentation
├── input.txt                             # Sample input for predictions
├── runP4.sh                              # Run prediction script
├── runConfig.sh                          # Run custom configuration training
└── runDefault.sh                         # Run default configuration training
```

## Requirements

- Java Development Kit (JDK) 8 or higher
- Bash shell (for running scripts)

## Usage

### Clone the Repository

To clone this repository, run:

```bash
git clone https://github.com/Machado65/AI-MNIST-2-3.git
cd AI-MNIST-2-3
```
### Training Models

**Default Configuration:**
```bash
./runDefault.sh
```

**Custom Configuration:**
```bash
./runConfig.sh
```

### Making Predictions

To classify digit images using a trained model:

```bash
./runP4.sh
```

This script:
1. Compiles all Java source files
2. Runs the P4 prediction program with input from `input.txt`
3. Outputs predicted labels (2 or 3)

The input format should be CSV with 400 pixel values per line (20×20 images).

## Architecture

### Neural Network Components

1. **MLP (Multi-Layer Perceptron)**
   - Configurable number of hidden layers
   - Supports multiple activation functions per layer
   - Implements backpropagation algorithm
   - Model save/load functionality

2. **Activation Functions**
   - **Sigmoid**: σ(z) = 1 / (1 + e^(-z))
   - **ReLU**: f(x) = max(0, x)
   - **Leaky ReLU**: f(x) = max(0.01x, x)
   - **Step**: Binary threshold function

3. **Data Augmentation**
   - Elastic deformation
   - Gaussian noise injection
   - Image rotation
   - Pixel shifting

### Training Process

1. Load and normalize MNIST data (digits 2 and 3)
2. Split into training and testing sets
3. Apply data augmentation techniques
4. Train MLP with backpropagation
5. Optimize classification threshold
6. Evaluate on test set
7. Save trained model

## Input Format

The prediction program expects CSV input where each line represents one 20×20 image:

```
pixel_1,pixel_2,pixel_3,...,pixel_400
```

Pixel values should be in the range [0, 255] or normalized [0, 1].

## Model Format

Trained models are saved in a custom text format containing:
- Network topology (layer sizes)
- Activation functions for each layer
- Weight matrices
- Bias vectors
- Optimal classification threshold

## Performance Optimization

The implementation includes several optimization techniques:
- Matrix operations optimized for neural network computations
- Efficient backpropagation algorithm
- Early stopping to prevent overfitting
- Threshold optimization for binary classification accuracy

## License

This project is an academic implementation for educational purposes.

## Notes

- The network specifically focuses on binary classification between digits 2 and 3
- Default model path: `src/ml/models/example.dat`
- Input size: 400 features (20×20 pixel images)
- Output: Binary classification (2 or 3)
