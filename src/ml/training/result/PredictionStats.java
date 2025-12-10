package ml.training.result;

import math.Matrix;

public class PredictionStats {
   private double min;
   private double max;
   private double avg;
   private int n;

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
