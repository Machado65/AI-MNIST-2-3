package neural;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import math.Array;
import math.Matrix;
import ml.training.config.TrainConfig;
import ml.training.result.TrainResult;
import ml.training.threshold.OptimalThreshold;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.ReLU;
import neural.activation.Sigmoid;
import neural.activation.Step;

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
   private Matrix[] dWPrev; // previous weight updates for momentum
   private Matrix[] dBPrev; // previous bias updates for momentum
   private final IDifferentiableFunction[] act; // activation functions for each layer
   private final int nLayers;
   private final int nLayers1;
   private OptimalThreshold optThreshold;
   private static final String WHITESPACE = "\\s+";

   private OptimalThreshold readOptimalThreshold(BufferedReader br)
         throws IOException {
      String line = br.readLine();
      if (!line.startsWith("OPTIMAL_THRESHOLD")) {
         throw new IOException(
               "Invalid model file: expected OPTIMAL_THRESHOLD");
      }
      return new OptimalThreshold(Double.parseDouble(
            line.split(WHITESPACE)[1]));
   }

   private int[] readTopology(BufferedReader br) throws IOException {
      String line = br.readLine();
      if (!line.startsWith("TOPOLOGY")) {
         throw new IOException(
               "Invalid model file: expected TOPOLOGY");
      }
      String[] topologyTok = line.substring(9)
            .split(WHITESPACE);
      int n = topologyTok.length;
      int[] layerSizes = new int[n];
      for (int i = 0; i < n; ++i) {
         layerSizes[i] = Integer.parseInt(topologyTok[i]);
      }
      return layerSizes;
   }

   private void readActivations(BufferedReader br) throws IOException {
      String line = br.readLine();
      if (!line.startsWith("ACTIVATIONS")) {
         throw new IOException(
               "Invalid model file: expected ACTIVATIONS");
      }
      String[] actTok = line.substring(12)
            .split(WHITESPACE);
      int n = actTok.length;
      for (int i = 0; i < n; ++i) {
         switch (actTok[i]) {
            case "Sigmoid":
               this.act[i] = new Sigmoid();
               break;
            case "ReLU":
               this.act[i] = new ReLU();
               break;
            case "LeakyReLU":
               this.act[i] = new LeakyReLU();
               break;
            case "Step":
               this.act[i] = new Step();
               break;
            default:
               throw new IOException("Unknown activation function: "
                     + actTok[i]);
         }
      }
   }

   private Matrix readMatrix(BufferedReader br) throws IOException {
      String[] dims = br.readLine().split(WHITESPACE);
      int rows = Integer.parseInt(dims[0]);
      int cols = Integer.parseInt(dims[1]);
      double[][] data = new double[rows][cols];
      for (int i = 0; i < rows; ++i) {
         String[] values = br.readLine().split(WHITESPACE);
         for (int j = 0; j < cols; ++j) {
            data[i][j] = Double.parseDouble(values[j]);
         }
      }
      return new Matrix(data);
   }

   private void readMatrices(BufferedReader br, String expectedHeader,
         Matrix[] matrices, int n) throws IOException {
      String line = br.readLine();
      if (line == null || !line.equals(expectedHeader)) {
         throw new IOException(
               "Invalid model file: expected " + expectedHeader);
      }
      for (int l = 0; l < n; ++l) {
         matrices[l] = readMatrix(br);
      }
   }

   public MLP(String path) throws IOException {
      try (BufferedReader br = new BufferedReader(new FileReader(path))) {
         this.optThreshold = readOptimalThreshold(br);
         int[] layerSizes = readTopology(br);
         this.nLayers = layerSizes.length;
         this.nLayers1 = this.nLayers - 1;
         this.yp = new Matrix[this.nLayers];
         this.w = new Matrix[this.nLayers1];
         this.b = new Matrix[this.nLayers1];
         this.act = new IDifferentiableFunction[this.nLayers1];
         readActivations(br);
         readMatrices(br, "WEIGHTS", this.w,
               this.nLayers1);
         readMatrices(br, "BIASES", this.b,
               this.nLayers1);
         String end = br.readLine();
         if (end == null || !end.equals("END")) {
            throw new IOException(
                  "Invalid model file: expected END");
         }
         this.dWPrev = new Matrix[this.nLayers1];
         this.dBPrev = new Matrix[this.nLayers1];
         for (int i = 0; i < this.nLayers1; ++i) {
            this.dWPrev[i] = new Matrix(this.w[i].rows(),
                  this.w[i].cols());
            this.dBPrev[i] = new Matrix(this.b[i].rows(),
                  this.b[i].cols());
         }
      }
   }

   /**
    * Constructs a Multi-Layer Perceptron with specified architecture.
    * Allows choosing between Xavier (for Sigmoid/Tanh) and He (for ReLU)
    * initialization.
    *
    * @param layerSizes array defining the number of neurons in each layer
    * @param act        array of activation functions for each layer
    * @param rand       random instance for weight initialization
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
         int n = layerSizes[i + 1];
         this.w[i] = Matrix.randHe(layerSizes[i], n, rand);
         double[][] biasData = new double[1][n];
         for (int j = 0; j < n; ++j) {
            // small random values in [-0.01, 0.01] for sigmoid
            biasData[0][j] = (rand.nextDouble() * 2.0 - 1.0) * 0.01;
         }
         this.b[i] = new Matrix(biasData);
      }
      this.dWPrev = new Matrix[this.nLayers1];
      this.dBPrev = new Matrix[this.nLayers1];
      for (int i = 0; i < this.nLayers1; ++i) {
         this.dWPrev[i] = new Matrix(this.w[i].rows(),
               this.w[i].cols());
         this.dBPrev[i] = new Matrix(this.b[i].rows(),
               this.b[i].cols());
      }
      // default threshold
      this.optThreshold = new OptimalThreshold(0.5);
   }

   /**
    * @return the optimal threshold used for classification.
    */
   public OptimalThreshold getOptimalThreshold() {
      return this.optThreshold;
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
      for (int i = 0; i < this.nLayers1; ++i) {
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
      for (int i = 0; i < this.nLayers1; ++i) {
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
         this.yp[l + 1] = this.yp[l].dot(this.w[l])
               .addRowVector(this.b[l]).apply(this.act[l].fnc());
      }
      return this.yp[this.nLayers - 1];
   }

   private Matrix computeDeltaForLayer(int l, int l1, Matrix e) {
      // delta = e .* yp[l+1] .* (1-yp[l+1])
      return e.mult(this.yp[l1].apply(this.act[l].derivative()));
   }

   private Matrix clipGradient(Matrix g, double clipNorm) {
      double norm = Math.sqrt(g.mult(g).sum());
      if (norm > clipNorm && norm > 0.0) {
         return g.mult(clipNorm / norm);
      }
      return g;
   }

   private void updateLayerSGD(int l, Matrix delta, double lr,
         double mom, double clipNorm) {
      Matrix deltaW = this.yp[l].transpose().dot(delta).mult(lr)
            .add(this.dWPrev[l].mult(mom));
      Matrix deltaB = delta.sumColumns().mult(lr)
            .add(this.dBPrev[l].mult(mom));
      deltaW = clipGradient(deltaW, clipNorm);
      deltaB = clipGradient(deltaB, clipNorm);
      this.w[l].addInPlace(deltaW);
      this.b[l].addInPlaceRowVector(deltaB);
      this.dWPrev[l] = deltaW;
      this.dBPrev[l] = deltaB;
   }

   public Matrix backPropagationSGD(Matrix y, double lr, double mom,
         double clipNorm) {
      int n = this.nLayers - 2;
      int n1 = n + 1;
      Matrix e = y.sub(this.yp[n1]);
      Matrix eOut = new Matrix(e);
      Matrix delta = this.computeDeltaForLayer(n, n1, e);
      this.updateLayerSGD(n, delta, lr, mom, clipNorm);
      for (int l = n - 1; l >= 0; --l) {
         int l1 = l + 1;
         e = delta.dot(this.w[l1].transpose());
         delta = this.computeDeltaForLayer(l, l1, e);
         this.updateLayerSGD(l, delta, lr, mom, clipNorm);
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
   public double[] trainSGD(Matrix x, Matrix y, double learningRate,
         int epochs, double mom) {
      int nSamples = x.rows();
      double[] mse = new double[epochs];
      for (int epoch = 0; epoch < epochs; ++epoch) {
         // forward propagation
         predict(x);
         // backward propagation
         Matrix e = backPropagationSGD(y, learningRate, mom,
               1.0);
         // mse
         mse[epoch] = e.mult(e).sum() / nSamples;
      }
      return mse;
   }

   private double computeOneCycleLR(double initLR, int epoch,
         int maxEpochs, double pctUp) {
      double lrLow = initLR / 10.0;
      double lrHigh = initLR * 3.0;
      double lrFinal = initLR / 100.0;
      double t = epoch / (double) maxEpochs;
      if (t < pctUp) {
         // warm-up phase
         return lrLow + (t / pctUp) * (lrHigh - lrLow);
      } else {
         // cool-down phase
         return lrHigh - (t - pctUp) / (1.0 - pctUp)
               * (lrHigh - lrFinal);
      }
   }

   /**
    * @param x
    * @param y
    * @param learningRate
    * @param epochs
    * @param patience
    * @return
    */
   public TrainResult train(TrainConfig config) {
      Matrix trX = config.getTr().getX();
      Matrix trY = config.getTr().getY();
      Matrix teX = config.getTe().getX();
      Matrix teY = config.getTe().getY();
      double learningRate = config.getLearningRate();
      int epochs = config.getEpochs();
      int patience = config.getPatience();
      int nTrain = trX.rows();
      int nTest = teX.rows();
      int noImprove = 0;
      int bestEpoch = 0;
      double bestTestMSE = Double.MAX_VALUE;
      double[] trainMSE = new double[epochs];
      double[] testMSE = new double[epochs];
      Matrix[] bestW = this.getWeightsCopy();
      Matrix[] bestB = this.getBiasesCopy();
      double minDelta = 1e-4;
      double pctUp = 0.3; // 30% warm-up, 70% cool-down
      double nom = 0.9;
      double clipNorm = 2.0;
      Array arr = new Array(nTrain);
      arr.initSequential(nTrain);
      for (int epoch = 0; epoch < epochs; ++epoch) {
         double lr = this.computeOneCycleLR(learningRate, epoch, epochs,
               pctUp);
         arr.shuffleArray(config.getRand());
         int batchSize = Math.min(64 + (epoch / 10) * 32, 256);
         for (int start = 0; start < nTrain; start += batchSize) {
            int end = Math.min(start + batchSize, nTrain);
            predict(trX.rows(arr, start, end));
            backPropagationSGD(trY.rows(arr, start, end), lr, nom,
                  clipNorm);
         }
         Matrix eTr = trY.sub(predict(trX));
         trainMSE[epoch] = eTr.mult(eTr).sum() / nTrain;
         Matrix eTe = teY.sub(predict(teX));
         testMSE[epoch] = eTe.mult(eTe).sum() / nTest;
         // early stopping check
         if (testMSE[epoch] < bestTestMSE - minDelta) {
            bestTestMSE = testMSE[epoch];
            noImprove = 0;
            bestEpoch = epoch;
            bestW = this.getWeightsCopy();
            bestB = this.getBiasesCopy();
         } else {
            ++noImprove;
            if (noImprove >= patience) {
               break;
            }
         }
      }
      this.w = bestW;
      this.b = bestB;
      int n = bestEpoch + 1;
      return new TrainResult(Arrays.copyOf(trainMSE, n),
            Arrays.copyOf(testMSE, n), bestEpoch, bestTestMSE);
   }

   private void writeOptimalThreshold(BufferedWriter bw,
         OptimalThreshold o) {
      try {
         bw.append("OPTIMAL_THRESHOLD ");
         bw.append(String.format("%.2f", o.getThreshold()));
         bw.newLine();
      } catch (Exception e) {
         e.printStackTrace();
      }
   }

   private int getLayerSize(int layerIdx) {
      return (layerIdx == 0) ? w[0].rows() : w[layerIdx - 1].cols();
   }

   private void writeTopology(BufferedWriter bw) throws IOException {
      bw.append("TOPOLOGY");
      for (int i = 0; i < nLayers; ++i) {
         bw.append(" ").append(Integer.toString(getLayerSize(i)));
      }
      bw.newLine();
   }

   private void writeActivations(BufferedWriter bw) throws IOException {
      bw.append("ACTIVATIONS");
      for (IDifferentiableFunction f : act) {
         bw.append(" ").append(f.getClass().getSimpleName());
      }
      bw.newLine();
   }

   private void writeMatrices(BufferedWriter bw, String label,
         Matrix[] matrices) throws IOException {
      bw.append(label);
      bw.newLine();
      for (Matrix matrix : matrices) {
         this.writeMatrix(bw, matrix);
      }
   }

   private void writeMatrix(BufferedWriter bw, Matrix matrix)
         throws IOException {
      bw.append(matrix.rows() + " " + matrix.cols());
      bw.newLine();
      int n = matrix.rows();
      int m = matrix.cols();
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < m; ++j) {
            bw.append(String.format("%.16f", matrix.get(i, j)))
                  .append(" ");
         }
         bw.newLine();
      }
   }

   public void saveModel(String path, OptimalThreshold o)
         throws IOException {
      try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
         this.writeOptimalThreshold(bw, o);
         this.writeTopology(bw);
         this.writeActivations(bw);
         this.writeMatrices(bw, "WEIGHTS", this.w);
         this.writeMatrices(bw, "BIASES", this.b);
         bw.append("END");
      }
   }
}
