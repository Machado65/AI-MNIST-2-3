package ml.training;

import java.util.Random;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;

public class Trainer {
   private double lr;
   private int epochs;
   private int patience;
   private MLP mlp;

   public Trainer(int[] topology, double lr, int epochs, int patience,
         IDifferentiableFunction[] act, Random rand) {
      this.lr = lr;
      this.epochs = epochs;
      this.patience = patience;
      this.mlp = new MLP(topology, act, rand);
   }

   public TrainResult train(Matrix trX, Matrix trY, Matrix teX, Matrix teY) {
      return mlp.train(trX, trY, teX, teY, lr, epochs, patience);
   }

   public Matrix predict(Matrix x) {
      return mlp.predict(x);
   }

   private void printPredictionDistribution(Matrix predictions) {
      int n = predictions.rows();
      double minPred = Double.MAX_VALUE;
      double maxPred = Double.MIN_VALUE;
      double sumPred = 0.0;
      for (int i = 0; i < n; i++) {
         double pred = predictions.get(i, 0);
         minPred = Math.min(minPred, pred);
         maxPred = Math.max(maxPred, pred);
         sumPred += pred;
      }
      System.out.println("\n=== Prediction Distribution ===");
      System.out.println("Min prediction: " + String.format("%.6f", minPred));
      System.out.println("Max prediction: " + String.format("%.6f", maxPred));
      System.out.println("Avg prediction: " + String.format("%.6f", sumPred / n));
   }

   private double findOptimalThreshold(Matrix predictions, Matrix actual) {
      double bestThreshold = 0.5;
      double bestAccuracy = 0.0;
      int n = predictions.rows();
      for (double threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
         int correct = countCorrectPredictions(predictions, actual, threshold);
         double accuracy = (double) correct / n;
         if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestThreshold = threshold;
         }
      }
      System.out.println("\n=== Threshold Optimization ===");
      System.out.println("Best threshold: " + String.format("%.2f", bestThreshold));
      System.out.println("Best accuracy: " + String.format("%.2f%%", bestAccuracy * 100));
      return bestThreshold;
   }

   private int countCorrectPredictions(Matrix predictions, Matrix actual, double threshold) {
      int correct = 0;
      int n = predictions.rows();
      for (int i = 0; i < n; i++) {
         int predClass = (predictions.get(i, 0) >= threshold) ? 1 : 0;
         int actualClass = (int) actual.get(i, 0);
         if (predClass == actualClass) {
            correct++;
         }
      }
      return correct;
   }

   private EvaluationMetrics computeMetrics(Matrix predictions, Matrix actual, double threshold) {
      EvaluationMetrics metrics = new EvaluationMetrics();
      int n = predictions.rows();
      for (int i = 0; i < n; i++) {
         int predClass = (predictions.get(i, 0) >= threshold) ? 1 : 0;
         int actualClass = (int) actual.get(i, 0);
         metrics.update(predClass, actualClass);
      }
      return metrics;
   }

   private void printEvaluationResults(EvaluationMetrics metrics) {
      System.out.println("\n=== Evaluation Metrics ===");
      System.out.println("Accuracy:  " + String.format("%.2f%%", metrics.getAccuracy() * 100));
      System.out.println("Precision: " + String.format("%.2f%%", metrics.getPrecision() * 100));
      System.out.println("Recall:    " + String.format("%.2f%%", metrics.getRecall() * 100));
      System.out.println("F1-Score:  " + String.format("%.4f", metrics.getF1Score()));
      System.out.println("\nConfusion Matrix:");
      System.out.println("              Predicted 2  Predicted 3");
      System.out.println("Actual 2:     " + metrics.trueNegatives + "            " + metrics.falsePositives);
      System.out.println("Actual 3:     " + metrics.falseNegatives + "            " + metrics.truePositives);
   }

   private static class EvaluationMetrics {
      int truePositives = 0;
      int falsePositives = 0;
      int falseNegatives = 0;
      int trueNegatives = 0;

      void update(int predicted, int actual) {
         if (predicted == actual) {
            if (actual == 1) {
               truePositives++;
            } else {
               trueNegatives++;
            }
         } else {
            if (predicted == 1) {
               falsePositives++;
            } else {
               falseNegatives++;
            }
         }
      }

      double getAccuracy() {
         int total = truePositives + trueNegatives + falsePositives + falseNegatives;
         return total > 0 ? (double) (truePositives + trueNegatives) / total : 0.0;
      }

      double getPrecision() {
         int predicted = truePositives + falsePositives;
         return predicted > 0 ? (double) truePositives / predicted : 0.0;
      }

      double getRecall() {
         int actualPositives = truePositives + falseNegatives;
         return actualPositives > 0 ? (double) truePositives / actualPositives : 0.0;
      }

      double getF1Score() {
         double precision = getPrecision();
         double recall = getRecall();
         return (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
      }
   }

   public void evaluate(Matrix teX, Matrix teY) {
      Matrix predictions = mlp.predict(teX);
      printPredictionDistribution(predictions);
      double bestThreshold = findOptimalThreshold(predictions, teY);
      EvaluationMetrics metrics = computeMetrics(predictions, teY, bestThreshold);
      printEvaluationResults(metrics);
   }
}
