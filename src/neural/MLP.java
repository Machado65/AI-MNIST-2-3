package neural;

import java.util.Arrays;
import java.util.Random;

import math.Matrix;
import ml.training.TrainResult;
import neural.activation.IDifferentiableFunction;

/**
 * Multi-Layer Perceptron (MLP) neural network implementation.
 * This class provides a fully connected feedforward neural network with
 * backpropagation
 * training algorithm. It supports multiple hidden layers with customizable
 * activation functions.
 *
 * The network uses the generalized delta rule for weight updates and can train
 * on
 * batch datasets using gradient descent.
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511052038
 */
public class MLP {
   private Matrix[] w; // weights for each layer
   private Matrix[] b; // biases for each layer
   private Matrix[] yp; // outputs for each layer (y predicted)
   private final IDifferentiableFunction[] act; // activation functions for each layer
   private final int nLayers;
   private final int nLayers1;

   /**
    * Constructs a Multi-Layer Perceptron with specified architecture.
    * Initializes weights and biases with random values using the provided seed.
    *
    * @param layerSizes array defining the number of neurons in each layer.
    *                   First element is input size, last is output size.
    *                   Must have at least 2 elements (input and output layers).
    * @param act        array of activation functions for each layer (excluding
    *                   input layer).
    *                   Must have length equal to layerSizes.length - 1.
    * @param seed       random seed for weight initialization; if negative, uses
    *                   current time
    * @throws IllegalArgumentException implicitly if preconditions are violated
    *
    * @pre layerSizes.length >= 2
    * @pre act.length == layerSizes.length - 1
    */
   public MLP(int[] layerSizes, IDifferentiableFunction[] act,
         Random rand) {
      this.nLayers = layerSizes.length;
      this.nLayers1 = this.nLayers - 1;
      // setup activation by layer
      this.act = act;
      // create output storage for each layer but the input layer
      this.yp = new Matrix[this.nLayers];
      // create weights and biases for each layer
      // each row in w[l] represents the weights that are input
      this.w = new Matrix[this.nLayers1];
      this.b = new Matrix[this.nLayers1];
      for (int i = 0; i < this.nLayers1; ++i) {
         this.w[i] = Matrix.rand(layerSizes[i], layerSizes[i + 1], rand);
         this.b[i] = Matrix.rand(1, layerSizes[i + 1], rand);
      }
   }

   /**
    * Returns the weight matrices for all layers.
    * Each matrix w[i] contains weights connecting layer i to layer i+1.
    * Each row represents the incoming weights to a neuron in the next layer.
    *
    * @return array of weight matrices
    */
   public Matrix[] getWeights() {
      return this.w;
   }

   /**
    * @return copy of the weight matrices for all layers.
    */
   public Matrix[] getWeightsCopy() {
      Matrix[] copy = new Matrix[this.nLayers1];
      for (int i = 0; i < this.nLayers1; i++) {
         copy[i] = new Matrix(this.w[i]);
      }
      return copy;
   }

   /**
    * Returns the bias vectors for all layers.
    * Each bias vector b[i] is a row vector (1 x n matrix) containing biases
    * for neurons in layer i+1.
    *
    * @return array of bias matrices (row vectors)
    */
   public Matrix[] getBiases() {
      return this.b;
   }

   /**
    * @return copy of the bias vectors for all layers.
    */
   public Matrix[] getBiasesCopy() {
      Matrix[] copy = new Matrix[this.nLayers1];
      for (int i = 0; i < this.nLayers1; i++) {
         copy[i] = new Matrix(this.b[i]);
      }
      return copy;
   }

   /**
    * Performs forward propagation through the network to make predictions.
    * Also used during training to compute outputs for each layer.
    *
    * Algorithm:
    * - yp[0] = X (input)
    * - yp[l+1] = activation(yp[l] * w[l] + b[l]) for each layer l
    *
    * @param x input matrix where each row is a sample and each column is a feature
    * @return output matrix with predictions (yp[nLayers-1])
    */
   public Matrix predict(Matrix x) {
      this.yp[0] = x;
      for (int l = 0; l < this.nLayers1; ++l) {
         this.yp[l + 1] = this.yp[l].dot(this.w[l]).addRowVector(this.b[l])
               .apply(this.act[l].fnc());
      }
      return this.yp[this.nLayers - 1];
   }

   /**
    * Updates weights and biases for a specific layer using the delta rule.
    *
    * Computes:
    * - delta = e .* yp[l+1] .* derivative(yp[l+1])
    * - w[l] += yp[l]^T * delta * lr
    * - b[l] += sum(delta) * lr
    *
    * @param l     the layer index to update
    * @param l1    the next layer index (l + 1)
    * @param delta matrix to store computed delta values (modified in place)
    * @param e     error matrix for this layer
    * @param lr    learning rate
    */
   private void updateLayer(int l, int l1, Matrix delta, Matrix e,
         double lr) {
      // delta = e .* yp[l+1] .* (1-yp[l+1])
      delta.set(e.mult(this.yp[l1].apply(this.act[l].derivative())));
      // w[l] += yp[l]^T * delta * lr
      this.w[l].addInPlace(this.yp[l].transpose().dot(delta).mult(lr));
      // b[l] += sum(delta) * lr
      b[l].addInPlaceRowVector(delta.sumColumns().mult(lr));
   }

   /**
    * Performs backpropagation to compute and apply weight updates.
    * Uses the generalized delta rule to propagate errors backward through the
    * network
    * and update weights and biases for all layers.
    *
    * Algorithm:
    * 1. Compute error at output layer: e = y - yp[output]
    * 2. Update output layer weights and biases
    * 3. Propagate error backward: e = delta * w[l+1]^T
    * 4. Update each hidden layer from last to first
    *
    * @param y  target output matrix (ground truth labels)
    * @param lr learning rate for weight updates
    * @return error matrix from the first hidden layer
    */
   public Matrix backPropagation(Matrix y, double lr) {
      Matrix delta = new Matrix(1, 1);// dummy initialization
      // back propagation using generalized delta rule
      int n = this.nLayers - 2;
      int n1 = n + 1;
      // output layer
      // e = y - yp[l+1]
      Matrix e = y.sub(this.yp[n1]);
      Matrix eOut = e;
      this.updateLayer(n, n1, delta, e, lr);
      // hidden layers
      for (int l = n - 1; l >= 0; --l) {
         int l1 = l + 1;
         // e = delta * w[l+1]^T
         e = delta.dot(this.w[l1].transpose());
         this.updateLayer(l, l1, delta, e, lr);
      }
      return eOut;
   }

   /**
    * Trains the neural network using batch gradient descent with backpropagation.
    * Performs forward propagation, backpropagation, and weight updates for each
    * epoch.
    * Computes and stores Mean Squared Error (MSE) for each epoch.
    *
    * @param x            training input matrix where each row is a sample
    * @param y            training output matrix (target values) where each row
    *                     corresponds to a sample
    * @param learningRate learning rate for gradient descent (controls step size)
    * @param epochs       number of training iterations over the entire dataset
    * @return array of MSE values, one for each epoch
    */
   public double[] train(Matrix x, Matrix y, double learningRate,
         int epochs) {
      int nSamples = x.rows();
      double[] mse = new double[epochs];
      for (int epoch = 0; epoch < epochs; ++epoch) {
         // forward propagation
         predict(x);
         // backward propagation
         Matrix e = backPropagation(y, learningRate);
         // mse
         mse[epoch] = e.dot(e.transpose()).get(0, 0)
               / nSamples;
      }
      return mse;
   }

   /**
    *
    * @param x
    * @param y
    * @param learningRate
    * @param epochs
    * @param patience
    * @return
    */
   public TrainResult train(Matrix trX, Matrix trY, Matrix teX,
         Matrix teY, double learningRate, int epochs, int patience) {
      int nTrain = trX.rows();
      int nTest = teX.rows();
      int noImprove = 0;
      int bestEpoch = 0;
      double bestTestMSE = Double.MAX_VALUE;
      double[] trainMSE = new double[epochs];
      double[] testMSE = new double[epochs];
      Matrix[] bestW = null;
      Matrix[] bestB = null;
      for (int epoch = 0; epoch < epochs; ++epoch) {
         predict(trX);
         Matrix eTr = backPropagation(trY, learningRate);
         trainMSE[epoch] = eTr.dot(eTr.transpose()).get(0, 0)
               / nTrain;
         // evaluate on test set (no backprop!)
         Matrix eTe = teY.sub(predict(teX));
         testMSE[epoch] = eTe.dot(eTe.transpose()).get(0, 0)
               / nTest;
         if (epoch % 100 == 0 || epoch < 10) {
            System.out.printf("Epoch %d: Train MSE=%.6f, Test MSE=%.6f%s%n",
                  epoch, trainMSE[epoch], testMSE[epoch],
                  (epoch == bestEpoch ? " *" : ""));
         }
         // early stopping check
         if (testMSE[epoch] < bestTestMSE) {
            bestTestMSE = testMSE[epoch];
            noImprove = 0;
            bestEpoch = epoch;
            // save best weights and biases
            bestW = this.getWeightsCopy();
            bestB = this.getBiasesCopy();
         } else {
            ++noImprove;
            if (noImprove >= patience) {
               break;
            }
         }
      }
      // restore best weights and biases
      this.w = bestW;
      this.b = bestB;
      int n = bestEpoch + 1;
      System.out.printf("Best epoch: %d (MSE: %.6f)%n",
            bestEpoch, bestTestMSE);
      return new TrainResult(Arrays.copyOf(trainMSE, n),
            Arrays.copyOf(testMSE, n), bestEpoch, bestTestMSE);
   }
}
