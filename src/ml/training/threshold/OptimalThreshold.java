package ml.training.threshold;

public class OptimalThreshold {
   private final double threshold;

   public OptimalThreshold(double threshold) {
      this.threshold = threshold;
   }

   public double getThreshold() {
      return this.threshold;
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("=== Threshold Optimization ===\n");
      sb.append(String.format("Best threshold: %.2f%n",
            this.threshold));
      return sb.toString();
   }
}
