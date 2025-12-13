package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.config.DataSet;
import ml.training.config.TrainConfig;
import ml.training.core.Trainer;
import ml.training.result.EvaluationResult;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class DefaultMLPConfig {
   private static final String DATASET_PATH = "data/largeDataset.csv";
   private static final String LABELS_PATH = "data/largeLabels.csv";
   private static final long SEED = 42;

   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.addGaussianNoise(0.02, 1, RandomProvider.of(SEED));
      ds.addElasticDeformation(6.0, 2.0, 1,
            RandomProvider.of(SEED));
      ds.addCombinedAugmentation2(1, RandomProvider.of(SEED),
            5.0, 1);
      // ds.addCombinedAugmentation1(1, RandomProvider.of(SEED),
      // 6.0, 2.0, 5.0);
      ds.split(0.8, RandomProvider.of(SEED));
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(new int[] { 400, 256, 1 },
            new IDifferentiableFunction[] {
                  new LeakyReLU(),
                  new Sigmoid() },
            new TrainConfig(new DataSet(trX, trY), new DataSet(teX, teY),
                  0.002, 16000, 800,
                  RandomProvider.of(SEED)),
            RandomProvider.of(SEED));
      System.out.println(trainer.train());
      EvaluationResult evalResult = trainer.evaluate();
      System.out.println(evalResult);
      try {
         trainer.getMLP().saveModel("src/ml/models/model.dat",
               evalResult.getOptimalThreshold());
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
