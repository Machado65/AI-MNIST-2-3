package apps.problem4;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.TrainResult;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class MLPTwoThree {
   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder("dataset.csv",
            "labels.csv");
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.addGaussianNoise(0.05, 2,
            RandomProvider.fixed());
      ds.split(0.8, RandomProvider.fixed());
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(new int[] { trX.cols(), 128, 64, 1 },
            0.01, 20000, 1000,
            new IDifferentiableFunction[] {
                  new Sigmoid(),
                  new Sigmoid(),
                  new Sigmoid() },
            RandomProvider.fixed());
      TrainResult result = trainer.train(trX, trY, teX, teY);
      System.out.println("Best Epoch: " + result.getBestEpoch());
      System.out.println("Best Test MSE: " + result.getBestTestMSE());
      trainer.evaluate(teX, teY);
   }
}
