package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.EvaluationResult;
import ml.training.Trainer;
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
      ds.addElasticDeformation(6.0, 2.0, 1,
            RandomProvider.of(SEED));
      // ds.addCombinedAugmentation1(1, RandomProvider.of(SEED),
      // 6.0, 2.0, 5.0);
      ds.split(0.8, RandomProvider.of(SEED));
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(new int[] { 400, 48, 1 },
            0.002, 16000, 800,
            new IDifferentiableFunction[] {
                  new LeakyReLU(),
                  new Sigmoid() },
            RandomProvider.of(SEED));
      System.out.println(trainer.train(trX, trY, teX, teY));
      System.out.println(trainer.evaluate(teX, teY));
      EvaluationResult evalResult = trainer.evaluate(teX, teY);
      try {
         trainer.getMLP().saveModel("src/ml/models/mlp_model.dat",
               evalResult.getOptimalThreshold());
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
