# AI-MNIST-2-3

**AI-MNIST-2-3** is a Java project that implements a Multi-Layer Perceptron (MLP) neural network to distinguish handwritten digits **2** and **3** from the MNIST dataset. It includes tools to **train** the model and **predict** labels from CSV input.

## Features

* **Custom MLP in Java**: Fully connected neural network implemented from scratch.
* **Binary Classification**: Classifies MNIST digits **2 vs 3**.
* **Multiple Activations**: Sigmoid, ReLU, Leaky ReLU, Step.
* **Data Augmentation**: Rotation, elastic deformation, Gaussian noise, pixel shifting.
* **Training Utilities**:
  * Backpropagation with gradient descent  
  * Early stopping (patience)  
  * Threshold optimization for best binary decision
* **Model Handling**: Save and load trained models from disk.
* **Utilities**: Custom matrix library, CSV reader, and random provider.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Machado65/AI-MNIST-2-3.git
   cd AI-MNIST-2-3
   ```

2. **Compile the Java sources** (creates/uses the `bin/` folder):

   ```bash
   mkdir -p bin
   javac -d bin $(find src -name "*.java")
   ```

   > The shell scripts (`runDefault.sh`, `runConfig.sh`, `runP4.sh`) also compile the project automatically, so you can skip the manual `javac` step if you prefer.

## How to Run

### Train a Model

**Default configuration**:

```bash
./runDefault.sh
```

**Custom configuration**:

```bash
./runConfig.sh
```

These scripts will compile the project (if needed), load MNIST data for digits 2 and 3, apply augmentation, train the MLP, and save the trained model and results (e.g., in `mse_results/`).

### Make Predictions

1. Prepare an input CSV file like `input.txt`, where each line is a **20×20** image (400 pixels):

   ```text
   pixel_1,pixel_2,...,pixel_400
   ```

   Pixel values should be in `[0, 255]` or normalized `[0, 1]`.

2. Run the prediction script:

   ```bash
   ./runP4.sh
   ```

   This will compile the code (if needed), load a trained model, read `input.txt`, and output predicted labels (**2** or **3**).

## Credits

* **Authors**: André Martins, António Matoso, Tomás Machado  
* Based on foundational work by **hdaniel@ualg.pt**  
* Academic project showcasing a from-scratch Java MLP for MNIST digit classification (2 vs 3).
