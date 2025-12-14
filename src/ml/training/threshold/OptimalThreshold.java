package ml.training.threshold;

/**
 * Represents the optimal classification threshold for binary classification.
 * The threshold value determines when a prediction is classified as positive.
 * Optimized to maximize accuracy on the validation set.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class OptimalThreshold {
   private final double threshold;

   /**
    * Constructs an optimal threshold with the specified value.
    *
    * @param threshold the threshold value (typically in [0.1, 0.9])
    */
   public OptimalThreshold(double threshold) {
      this.threshold = threshold;
   }

   /**
    * Returns the optimal threshold value.
    *
    * @return the threshold value
    */
   public double getThreshold() {
      return this.threshold;
   }

   /**
    * Returns a formatted string with the threshold value.
    *
    * @return formatted string displaying the threshold
    */
   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("=== Threshold Optimization ===\n");
      sb.append(String.format("Best threshold: %.2f%n",
            this.threshold));
      return sb.toString();
   }
}
