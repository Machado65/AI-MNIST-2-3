package ml.training.core;

import java.util.Random;

import math.Matrix;
import ml.training.config.TrainConfig;
import ml.training.result.EvaluationMetrics;
import ml.training.result.EvaluationResult;
import ml.training.result.PredictionStats;
import ml.training.result.TrainResult;
import ml.training.threshold.OptimalThreshold;
import neural.MLP;
import neural.activation.IDifferentiableFunction;

/**
 * High-level trainer class for Multi-Layer Perceptron (MLP).
 * Manages the training lifecycle, prediction, and evaluation of the neural
 * network.
 * Provides automatic threshold optimization for binary classification.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class Trainer {
   TrainConfig config;
   private MLP mlp;

   /**
    * Constructs a trainer with specified network architecture and configuration.
    *
    * @param topology array defining layer sizes (e.g., [400, 64, 1])
    * @param act      array of activation functions for each layer
    * @param config   training configuration (datasets, hyperparameters)
    * @param rand     random number generator for reproducibility
    */
   public Trainer(int[] topology, IDifferentiableFunction[] act,
         TrainConfig config, Random rand) {
      this.config = new TrainConfig(config);
      this.mlp = new MLP(topology, act, rand);
   }

   /**
    * Returns the underlying MLP instance.
    *
    * @return the MLP network
    */
   public MLP getMLP() {
      return this.mlp;
   }

   /**
    * Trains the MLP using the configuration provided at construction.
    * Uses SGD with momentum, dropout, and early stopping.
    *
    * @return training result containing MSE history and best epoch
    */
   public TrainResult train() {
      return mlp.train(config);
   }

   /**
    * Makes predictions on the given input data.
    * Dropout is disabled during prediction.
    *
    * @param x input feature matrix (rows = samples, cols = features)
    * @return matrix of predictions (one probability per row)
    */
   public Matrix predict(Matrix x) {
      return mlp.predict(x);
   }

   /**
    * Counts correct predictions using the specified threshold.
    *
    * @param pred      predicted probabilities
    * @param actual    actual labels (0 or 1)
    * @param threshold classification threshold
    * @param n         number of samples
    * @return number of correct predictions
    */
   private int countCorrectPredictions(Matrix pred, Matrix actual,
         double threshold, int n) {
      int ans = 0;
      for (int i = 0; i < n; ++i) {
         if (((pred.get(i, 0) < threshold)
               ? 0
               : 1) == (int) actual.get(i, 0)) {
            ++ans;
         }
      }
      return ans;
   }

   /**
    * Finds the optimal classification threshold by grid search.
    * Tests thresholds from 0.1 to 0.9 in steps of 0.05.
    *
    * @param pred   predicted probabilities
    * @param actual actual labels
    * @param n      number of samples
    * @return optimal threshold that maximizes accuracy
    */
   private OptimalThreshold findOptimalThreshold(Matrix pred,
         Matrix actual, int n) {
      double bestThreshold = 0.5;
      double bestAccuracy = 0.0;
      for (double threshold = 0.1; threshold <= 0.90; threshold += 0.05) {
         double accuracy = (double) countCorrectPredictions(
               pred, actual, threshold, n) / n;
         if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestThreshold = threshold;
         }
      }
      return new OptimalThreshold(bestThreshold);
   }

   /**
    * Computes evaluation metrics (TP, FP, TN, FN) using the threshold.
    *
    * @param pred      predicted probabilities
    * @param actual    actual labels
    * @param threshold classification threshold
    * @param n         number of samples
    * @return evaluation metrics with accuracy, precision, recall, F1
    */
   private EvaluationMetrics computeMetrics(Matrix pred, Matrix actual,
         double threshold, int n) {
      EvaluationMetrics metrics = new EvaluationMetrics();
      for (int i = 0; i < n; ++i) {
         metrics.update((pred.get(i, 0) < threshold) ? 0 : 1,
               (int) actual.get(i, 0));
      }
      return metrics;
   }

   /**
    * Evaluates the model on the test dataset.
    * Automatically finds the optimal threshold and computes all metrics.
    *
    * @return complete evaluation result with statistics, threshold, and metrics
    */
   public EvaluationResult evaluate() {
      Matrix teX = config.getTe().getX();
      Matrix teY = config.getTe().getY();
      Matrix pred = predict(teX);
      int n = teX.rows();
      OptimalThreshold optimalThreshold = findOptimalThreshold(pred,
            teY, n);
      return new EvaluationResult(new PredictionStats(pred),
            optimalThreshold, computeMetrics(pred, teY,
                  optimalThreshold.getThreshold(), n));
   }
}
