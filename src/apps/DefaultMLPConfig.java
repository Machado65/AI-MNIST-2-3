package apps;

import java.util.Random;

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
   private static Random rand = RandomProvider.of(SEED);

   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.addElasticDeformation(6.0, 2.0, 1, rand);
      // ds.addCombinedAugmentation1(1, RandomProvider.of(SEED),
      // 6.0, 2.0, 5.0);
      ds.split(0.8, rand);
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      double lr = 0.002;
      int epochs = 16000;
      int patience = 800;
      Trainer trainer = new Trainer(new int[] { 400, 48, 1 },
            new IDifferentiableFunction[] {
                  new LeakyReLU(),
                  new Sigmoid() },
            new TrainConfig(new DataSet(trX, trY), new DataSet(teX, teY),
                  lr, epochs, patience, rand),
            rand);
      System.out.println(trainer.train());
      EvaluationResult evalResult = trainer.evaluate();
      System.out.println(evalResult);
      try {
         trainer.getMLP().saveModel("src/ml/models/mlp_model.dat",
               evalResult.getOptimalThreshold());
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
