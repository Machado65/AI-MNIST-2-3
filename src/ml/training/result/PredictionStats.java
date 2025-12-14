package ml.training.result;

import math.Matrix;

/**
 * Statistics about model predictions: minimum, maximum, and average values.
 * Useful for understanding the distribution of predicted probabilities.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class PredictionStats {
   private double min;
   private double max;
   private double avg;
   private int n;

   /**
    * Computes statistics from a matrix of predictions.
    *
    * @param pred matrix containing predicted probabilities (one per row)
    */
   public PredictionStats(Matrix pred) {
      this.n = pred.rows();
      this.min = Double.MAX_VALUE;
      this.max = Double.MIN_VALUE;
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
         double p = pred.get(i, 0);
         this.min = Math.min(this.min, p);
         this.max = Math.max(this.max, p);
         sum += p;
      }
      this.avg = sum / n;
   }

   /**
    * Returns a formatted string with prediction statistics.
    *
    * @return string containing count, min, max, and average predictions
    */
   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("=== Prediction Distribution ===\n");
      sb.append(String.format("Count: %d%n", n));
      sb.append(String.format("Min prediction: %.6f%n", min));
      sb.append(String.format("Max prediction: %.6f%n", max));
      sb.append(String.format("Avg prediction: %.6f", avg));
      return sb.toString();
   }
}
