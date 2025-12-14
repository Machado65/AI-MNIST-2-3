package ml.training.result;

/**
 * Computes and stores classification metrics for binary classification.
 * Tracks true positives, false positives, true negatives, and false negatives.
 * Calculates accuracy, precision, recall, and F1-score.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class EvaluationMetrics {
   private int truePos;
   private int falsePos;
   private int falseNeg;
   private int trueNeg;

   /**
    * Constructs a new EvaluationMetrics instance with all counts initialized to
    * zero.
    */
   public EvaluationMetrics() {
      this.truePos = 0;
      this.falsePos = 0;
      this.falseNeg = 0;
      this.trueNeg = 0;
   }

   /**
    * Returns the number of true positive predictions.
    *
    * @return true positive count
    */
   public int getTruePositives() {
      return this.truePos;
   }

   /**
    * Returns the number of false positive predictions.
    *
    * @return false positive count
    */
   public int getFalsePositives() {
      return this.falsePos;
   }

   /**
    * Returns the number of false negative predictions.
    *
    * @return false negative count
    */
   public int getFalseNegatives() {
      return this.falseNeg;
   }

   /**
    * Returns the number of true negative predictions.
    *
    * @return true negative count
    */
   public int getTrueNegatives() {
      return this.trueNeg;
   }

   /**
    * Updates the confusion matrix with a new prediction.
    *
    * @param predicted the predicted class (0 or 1)
    * @param actual    the actual class (0 or 1)
    */
   public void update(int predicted, int actual) {
      if (predicted == actual) {
         if (actual == 1) {
            ++truePos;
         } else {
            ++trueNeg;
         }
      } else {
         if (predicted == 1) {
            ++falsePos;
         } else {
            ++falseNeg;
         }
      }
   }

   /**
    * Calculates the accuracy: (TP + TN) / (TP + TN + FP + FN).
    *
    * @return accuracy in [0, 1], or 0.0 if no predictions
    */
   public double getAccuracy() {
      double total = (double) truePos + trueNeg + falsePos + falseNeg;
      return total > 0
            ? (truePos + trueNeg) / total
            : 0.0;
   }

   /**
    * Calculates the precision: TP / (TP + FP).
    * Measures the fraction of positive predictions that are correct.
    *
    * @return precision in [0, 1], or 0.0 if no positive predictions
    */
   public double getPrecision() {
      double pred = (double) truePos + falsePos;
      return pred > 0 ? truePos / pred : 0.0;
   }

   /**
    * Calculates the recall (sensitivity): TP / (TP + FN).
    * Measures the fraction of actual positives that are correctly identified.
    *
    * @return recall in [0, 1], or 0.0 if no actual positives
    */
   public double getRecall() {
      double actualPos = (double) truePos + falseNeg;
      return actualPos > 0 ? truePos / actualPos : 0.0;
   }

   /**
    * Calculates the F1-score: 2 * (precision * recall) / (precision + recall).
    * Harmonic mean of precision and recall.
    *
    * @return F1-score in [0, 1], or 0.0 if both precision and recall are 0
    */
   public double getFMeasure() {
      double p = getPrecision();
      double r = getRecall();
      return (p + r > 0) ? 2 * p * r / (p + r) : 0.0;
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("=== Evaluation Metrics ===\n");
      sb.append(String.format("Accuracy:  %.2f%%%n",
            getAccuracy() * 100));
      sb.append(String.format("Precision: %.2f%%%n",
            getPrecision() * 100));
      sb.append(String.format("Recall:    %.2f%%%n",
            getRecall() * 100));
      sb.append(String.format("F-Measure: %.4f%n",
            getFMeasure()));
      sb.append("Confusion Matrix:\n");
      sb.append(String.format("              Predicted 0  Predicted 1%n"));
      sb.append(String.format("Actual 0:     %d            %d%n",
            trueNeg, falsePos));
      sb.append(String.format("Actual 1:     %d            %d",
            falseNeg, truePos));
      return sb.toString();
   }
}
