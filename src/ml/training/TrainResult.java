package ml.training;

public class TrainResult {
   private double[] trainMSE;
   private double[] testMSE;
   private int bestEpoch;
   private double bestTestMSE;

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
}
