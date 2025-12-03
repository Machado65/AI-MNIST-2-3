package ml.training;

public class OptimalThreshold {
   private final double threshold;
   private final double accuracy;

   public OptimalThreshold(double threshold, double accuracy) {
      this.threshold = threshold;
      this.accuracy = accuracy;
   }

   public double getThreshold() {
      return this.threshold;
   }

   public double getAccuracy() {
      return this.accuracy;
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("=== Threshold Optimization ===\n");
      sb.append(String.format("Best threshold: %.2f%n",
            this.threshold));
      sb.append(String.format("Best accuracy:  %.2f%%",
            this.accuracy * 100));
      return sb.toString();
   }
}
