package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class DefaultMLPConfig {
   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder(
            "src/ml/data/dataset.csv",
            "src/ml/data/labels.csv");
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      // ds.normalize();
      // ds.addGaussianNoise(0.005, 1, RandomProvider.fixed());
      ds.split(0.8, RandomProvider.fixed());
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(new int[] { trX.cols(), 32, 1 },
            0.1, 10000, 200,
            new IDifferentiableFunction[] {
                  new Sigmoid(),
                  new Sigmoid() },
            RandomProvider.fixed());
      System.out.println(trainer.train(trX, trY, teX, teY));
      System.out.println(trainer.evaluate(teX, teY));
      try {
         trainer.getMLP().saveModel("src/ml/models/mlp_model.dat");
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
