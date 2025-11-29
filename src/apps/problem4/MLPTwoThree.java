package apps.problem4;

import ml.data.DataSetBuilder;
import ml.training.TrainResult;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class MLPTwoThree {
   public static void main(String[] args) {
      DataSetBuilder dl = new DataSetBuilder("dataset.csv",
            "labels.csv");
      dl.normalize();
      dl.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      dl.addGaussianNoise(0.05, 2,
            RandomProvider.fixed());
      dl.split(0.8, RandomProvider.fixed());
      Trainer trainer = new Trainer(new int[] { 400, 128, 64, 1 },
            0.01, 20000, 1000,
            new IDifferentiableFunction[] {
                  new Sigmoid(),
                  new Sigmoid(),
                  new Sigmoid() },
            RandomProvider.fixed());
      TrainResult result = trainer.train(dl.getTrX(), dl.getTrY(),
            dl.getTeX(), dl.getTeY());
      System.out.println("Best Epoch: " + result.getBestEpoch());
      System.out.println("Best Test MSE: " + result.getBestTestMSE());
      trainer.evaluate(dl.getTeX(), dl.getTeY());

      System.out.println("Dataset info:");
      System.out.println("Training samples: " + dl.getTrX().rows());
      System.out.println("Testing samples: " + dl.getTeX().rows());
      System.out.println("\nSample labels (first 10):");
      for (int i = 0; i < Math.min(10, dl.getTrY().rows()); i++) {
         System.out.printf("Label[%d] = %.0f%n", i, dl.getTrY().get(i, 0));
      }
   }
}
