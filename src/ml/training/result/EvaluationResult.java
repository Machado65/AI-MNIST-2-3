package ml.training.result;

import ml.training.threshold.OptimalThreshold;

/**
 * Aggregates all evaluation results: prediction statistics, optimal threshold,
 * and metrics.
 * Immutable container for comprehensive model evaluation results.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class EvaluationResult {
   public final PredictionStats p;
   public final OptimalThreshold o;
   public final EvaluationMetrics m;

   /**
    * Constructs an evaluation result with all components.
    *
    * @param p prediction statistics (min, max, avg)
    * @param o optimal classification threshold
    * @param m evaluation metrics (accuracy, precision, recall, F1)
    */
   public EvaluationResult(PredictionStats p, OptimalThreshold o,
         EvaluationMetrics m) {
      this.p = p;
      this.o = o;
      this.m = m;
   }

   /**
    * Returns the optimal classification threshold.
    *
    * @return the optimal threshold
    */
   public OptimalThreshold getOptimalThreshold() {
      return this.o;
   }

   /**
    * Returns the accuracy from the evaluation metrics.
    *
    * @return the accuracy value
    */
   public double getAccuracy() {
      return this.m.getAccuracy();
   }

   /**
    * Returns a string representation containing all evaluation results.
    *
    * @return formatted string with predictions, threshold, and metrics
    */
   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(this.p).append("\n");
      sb.append(this.o).append("\n");
      sb.append(this.m);
      return sb.toString();
   }
}
