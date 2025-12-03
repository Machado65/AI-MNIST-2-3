import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.EvaluationResult;
import ml.training.TrainResult;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class P4 {
   private static final int INPUT_SIZE = 400;

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
      TrainResult trainResult = trainer.train(trX, trY, teX, teY);
      EvaluationResult evalResult = trainer.evaluate(teX, teY);
      try (BufferedReader br = new BufferedReader(
            new InputStreamReader(System.in))) {
         String line;
         List<Matrix> pred = new ArrayList<>();
         while ((line = br.readLine()) != null) {
            if (line.equals("-1")) {
               System.out.println(trainResult);
               System.out.println(evalResult);
               break;
            } else {
               String[] values = line.split(",");
               double[][] input = new double[1][INPUT_SIZE];
               for (int i = 0; i < INPUT_SIZE; ++i) {
                  input[0][i] = Double.parseDouble(values[i]);
               }
               pred.add(trainer.predict(new Matrix(input))
                     .apply(v -> (v < evalResult
                           .getOptimalThreshold().getThreshold())
                                 ? 2
                                 : 3));
            }
         }
         for (Matrix p : pred) {
            System.out.print(p.toIntString());
         }
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
}
