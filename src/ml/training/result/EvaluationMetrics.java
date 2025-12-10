package ml.training.result;

public class EvaluationMetrics {
   private int truePos;
   private int falsePos;
   private int falseNeg;
   private int trueNeg;

   public EvaluationMetrics() {
      this.truePos = 0;
      this.falsePos = 0;
      this.falseNeg = 0;
      this.trueNeg = 0;
   }

   public int getTruePositives() {
      return this.truePos;
   }

   public int getFalsePositives() {
      return this.falsePos;
   }

   public int getFalseNegatives() {
      return this.falseNeg;
   }

   public int getTrueNegatives() {
      return this.trueNeg;
   }

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

   public double getAccuracy() {
      double total = (double) truePos + trueNeg + falsePos + falseNeg;
      return total > 0
            ? (truePos + trueNeg) / total
            : 0.0;
   }

   public double getPrecision() {
      double pred = (double) truePos + falsePos;
      return pred > 0 ? truePos / pred : 0.0;
   }

   public double getRecall() {
      double actualPos = (double) truePos + falseNeg;
      return actualPos > 0 ? truePos / actualPos : 0.0;
   }

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
