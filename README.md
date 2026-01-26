# AI-MNIST-2-3

A Java implementation of a Multi-Layer Perceptron (MLP) neural network for binary classification of MNIST handwritten digits 2 and 3.

## Overview

This project implements a fully connected feedforward neural network with backpropagation training for distinguishing between handwritten digits 2 and 3 from the MNIST dataset. The implementation includes custom matrix operations, multiple activation functions, and data augmentation techniques.

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

> **⚠️ Disclaimer:** The `MLPConfig` and `DefaultMLPConfig` training processes may take some time to complete due to the neural network training and optimization procedures.

#### **runConfig.sh - Hyperparameter Search**

```bash
./runConfig.sh
```

The `runConfig.sh` script executes the **MLPConfig** program, which is used for **hyperparameter search and configuration optimization**. This process:

**Purpose:** Find the best MLP configuration through systematic experimentation.

**Characteristics:**
- **Multiple Seeds**: Tests various random seeds (42, 97, 123, 456, 789, 1337, 2023, 9999, 314159, 271828, etc.) to ensure result robustness
- **Multiple Configurations**: Runs different network configurations (config0, config1, config2, etc.) with:
  - Different architectures (number of neurons per layer)
  - Different learning rates
  - Different data augmentation techniques
  - Different epochs and patience for early stopping
- **Varied Augmentations**: Each configuration tests different combinations of:
  - Gaussian Noise
  - Elastic Deformation
  - Image Rotation
  - Combined Augmentations (1, 2, 3)
- **Result Comparison**: Allows comparing multiple models to identify the best one

**Example Configuration (config1):**
```java
- Architecture: [400, 256, 1] (input layer, hidden layer, output layer)
- Learning rate: 0.002
- Max epochs: 16000
- Patience: 800
- Activation functions: LeakyReLU (hidden layer), Sigmoid (output)
- Augmentations: CombinedAugmentation2 + CombinedAugmentation3
```

**When to use:**
- When you want to find the best configuration for your problem
- During experimentation and development phase
- To test new augmentation techniques or architectures

---

#### **runDefault.sh - Optimized Training**

```bash
./runDefault.sh
```

The `runDefault.sh` script executes the **DefaultMLPConfig** program, which uses the **best configuration already found** during the search phase with `runConfig.sh`.

**Purpose:** Train a final model with the optimized configuration.

**Characteristics:**
- **Single Configuration**: Uses only one pre-determined configuration (the best found)
- **Fixed Seed**: Uses a specific seed (2023) for reproducibility
- **Optimized Augmentations**: Applies the best augmentation techniques identified:
  ```java
  - CombinedAugmentation1: Gaussian noise (0.02), scaling (0.9-1.1)
  - CombinedAugmentation2: Elastic deformation (sigma=6.0, repetitions=1)
  - CombinedAugmentation3: Elastic deformation (sigma=6.0, alpha=2.0)
  ```
- **Optimized Architecture**:
  ```java
  - Layers: [400, 256, 1]
  - Learning rate: 0.001
  - Max epochs: 16000
  - Patience: 800
  - Activations: LeakyReLU + Sigmoid
  ```
- **Model Saving**: Saves the trained model to `src/ml/models/model.dat`
- **Metrics Saving**: Saves MSE (Mean Squared Error) values to `mse_results/`

**When to use:**
- When you want to quickly train the final model
- For production or final evaluation
- When configurations have already been optimized

---

### Comparison: runConfig.sh vs runDefault.sh

| Aspect | **runConfig.sh** | **runDefault.sh** |
|---------|------------------|-------------------|
| **Program** | `apps.MLPConfig` | `apps.DefaultMLPConfig` |
| **Purpose** | Hyperparameter search | Training with best configuration |
| **Seeds** | Multiple (14+ seeds) | Single (2023) |
| **Configurations** | Multiple (config0-9) | One optimized |
| **Execution time** | Very long (hours) | Moderate (minutes) |
| **Output** | Comparison of multiple models | One final model |
| **Use case** | Experimentation and development | Production |

---

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
   - Combined augmentations (1, 2, 3)

### Training Process

1. Load and normalize MNIST data (digits 2 and 3)
2. Split into training and testing sets
3. Apply data augmentation techniques
4. Train MLP with backpropagation
5. Optimize classification threshold
6. Evaluate on test set
7. Save trained model

## Configuration Details

### TrainConfig Parameters

The `TrainConfig` class contains the following parameters:

- **Training Dataset** (`DataSet tr`): Features and labels for training
- **Testing Dataset** (`DataSet te`): Features and labels for validation
- **Learning Rate** (`double learningRate`): Step size for gradient descent (e.g., 0.001, 0.002)
- **Epochs** (`int epochs`): Maximum number of training iterations (typically 16000)
- **Patience** (`int patience`): Early stopping threshold (typically 800)
- **Random Seed** (`Random rand`): For reproducible results

### Augmentation Techniques

1. **Gaussian Noise**: Adds random noise with standard deviation σ
2. **Elastic Deformation**: Applies smooth distortions to images
3. **Rotation**: Rotates images by random angles
4. **Combined Augmentation 1**: Noise + scaling
5. **Combined Augmentation 2**: Elastic deformation (single pass)
6. **Combined Augmentation 3**: Elastic deformation (double pass)

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
- Default model path: `src/ml/models/model.dat`
- Input size: 400 features (20×20 pixel images)
- Output: Binary classification (2 or 3)
