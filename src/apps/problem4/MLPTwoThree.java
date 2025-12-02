package apps.problem4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class MLPTwoThree {
   public static void main(String[] args) {
      DataSetBuilder ds = new DataSetBuilder("dataset.csv",
            "labels.csv");
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
      trainer.train(trX, trY, teX, teY);
      try (BufferedReader br = new BufferedReader(
            new InputStreamReader(System.in))) {
         String line = br.readLine();
         if (line != null && !line.isEmpty()) {
            if (line.equals("-1")) {
               trainer.evaluate(teX, teY);
            } else {
               String[] values = line.split(",");
               double[][] input = new double[1][400];
               for (int i = 0; i < 400; ++i) {
                  input[0][i] = Double.parseDouble(values[i]);
               }
               Matrix pred = trainer.predict(new Matrix(input))
                     .apply(v -> (v < 0.5) ? 2 : 3);
               System.out.print(pred.toIntString());
            }
         }
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
}
