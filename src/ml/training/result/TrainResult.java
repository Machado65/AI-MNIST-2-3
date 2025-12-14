package ml.training.result;

/**
 * Contains training results including MSE history and best epoch information.
 * Tracks both training and testing MSE across epochs.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class TrainResult {
   private final double[] trainMSE;
   private final double[] testMSE;
   private final int bestEpoch;
   private final double bestTestMSE;

   /**
    * Constructs a training result with complete epoch history.
    *
    * @param trainMSE    array of training MSE values per epoch
    * @param testMSE     array of testing MSE values per epoch
    * @param bestEpoch   epoch number with lowest test MSE
    * @param bestTestMSE the lowest test MSE achieved
    */
   public TrainResult(double[] trainMSE, double[] testMSE,
         int bestEpoch, double bestTestMSE) {
      this.trainMSE = trainMSE;
      this.testMSE = testMSE;
      this.bestEpoch = bestEpoch;
      this.bestTestMSE = bestTestMSE;
   }

   /**
    * Returns the training MSE history.
    *
    * @return array of training MSE per epoch
    */
   public double[] getTrainMSE() {
      return this.trainMSE;
   }

   /**
    * Returns the testing MSE history.
    *
    * @return array of testing MSE per epoch
    */
   public double[] getTestMSE() {
      return this.testMSE;
   }

   /**
    * Returns the epoch with the best (lowest) test MSE.
    *
    * @return the best epoch number
    */
   public int getBestEpoch() {
      return this.bestEpoch;
   }

   /**
    * Returns the lowest test MSE achieved during training.
    *
    * @return the best test MSE
    */
   public double getBestTestMSE() {
      return this.bestTestMSE;
   }

   /**
    * Returns a summary of training results.
    *
    * @return formatted string with best epoch and MSE
    */
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
