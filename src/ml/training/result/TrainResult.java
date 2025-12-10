package ml.training.result;

public class TrainResult {
   private final double[] trainMSE;
   private final double[] testMSE;
   private final int bestEpoch;
   private final double bestTestMSE;

   public TrainResult(double[] trainMSE, double[] testMSE,
         int bestEpoch, double bestTestMSE) {
      this.trainMSE = trainMSE;
      this.testMSE = testMSE;
      this.bestEpoch = bestEpoch;
      this.bestTestMSE = bestTestMSE;
   }

   public double[] getTrainMSE() {
      return this.trainMSE;
   }

   public double[] getTestMSE() {
      return this.testMSE;
   }

   public int getBestEpoch() {
      return this.bestEpoch;
   }

   public double getBestTestMSE() {
      return this.bestTestMSE;
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("===Training Results===\n");
      sb.append("Best Epoch: ").append(bestEpoch).append("\n");
      sb.append(String.format("Best Test MSE: %.6f",
            bestTestMSE));
      return sb.toString();
   }
}
