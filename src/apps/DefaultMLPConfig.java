package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.config.DataSet;
import ml.training.config.TrainConfig;
import ml.training.core.Trainer;
import ml.training.result.EvaluationResult;
import ml.training.result.TrainResult;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import utils.MSE;
import utils.RandomProvider;

public class DefaultMLPConfig {
   private static final String DATASET_PATH = "data/mediumDataset.csv";
   private static final String LABELS_PATH = "data/mediumLabels.csv";
   private static final long SEED = 2023;

   /**
    * Main method that trains the MLP with default configuration.
    * Loads data, applies augmentation, trains the network, and evaluates
    * performance.
    * Saves the trained model to disk.
    *
    * @param args command line arguments (not used)
    */
   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      // ds.addGaussianNoise(0.02, 1,
      // RandomProvider.of(SEED));
      // ElasticDeformation(6.0, 2.0, 1,
      // RandomProvider.of(SEED));
      // ds.addRotation(5.0, 1,
      // RandomProvider.of(SEED));
      ds.addCombinedAugmentation1(1, RandomProvider.of(SEED),
            0.02, 0.9, 1.1, 0.9,
            1.1);
      ds.addCombinedAugmentation2(1, RandomProvider.of(SEED),
            6.0, 1);
      ds.addCombinedAugmentation3(1, RandomProvider.of(SEED),
            6.0, 2.0);
      ds.split(0.8, RandomProvider.of(SEED));
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(
            new int[] { 400, 256, 1 },
            new IDifferentiableFunction[] {
                  new LeakyReLU(),
                  new Sigmoid() },
            new TrainConfig(
                  new DataSet(trX, trY),
                  new DataSet(teX, teY),
                  0.001,
                  16000,
                  800,
                  RandomProvider.of(SEED)),
            RandomProvider.of(SEED));
      TrainResult trainResult = trainer.train();
      System.out.println(trainResult);
      EvaluationResult evalResult = trainer.evaluate();
      System.out.println(evalResult);
      try {
         trainer.getMLP().saveModel(
               "src/ml/models/model.dat",
               evalResult.getOptimalThreshold());
         MSE.saveMSE(trainResult.getTrainMSE(),
               "mse_results/train.csv");
         MSE.saveMSE(trainResult.getTestMSE(),
               "mse_results/test.csv");
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
