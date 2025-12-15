package apps;

import java.util.Scanner;

import math.Matrix;
import ml.training.config.DataSet;
import ml.training.config.TrainConfig;
import ml.training.core.Trainer;
import ml.training.result.EvaluationResult;
import ml.training.result.TrainResult;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511050300
 */
public class MLPNXOR {
   public static void main(String[] args) {
      // Dataset
      Matrix trX = new Matrix(
            new double[][] {
                  { 0, 0 },
                  { 0, 1 },
                  { 1, 0 },
                  { 1, 1 } });
      Matrix trY = new Matrix(
            new double[][] {
                  { 1 },
                  { 0 },
                  { 0 },
                  { 1 } });
      TrainConfig config = new TrainConfig(
            new DataSet(trX, trY),
            new DataSet(trX, trY),
            0.1,
            50000,
            10000,
            RandomProvider.of(42));
      Trainer trainer = new Trainer(
            new int[] { 2, 2, 1 },
            new IDifferentiableFunction[] {
                  new Sigmoid(),
                  new Sigmoid() },
            config,
            RandomProvider.of(42));
      TrainResult trainResult = trainer.train();
      System.out.println("Train Result: " + trainResult);
      EvaluationResult evalResult = trainer.evaluate();
      System.out.println("Evaluation Result: " + evalResult);
      Scanner sc = new Scanner(System.in);
      System.out.println("Insira o número de inputs:");
      int n = sc.nextInt();
      double[][] userInput = new double[n][2];
      for (int i = 0; i < n; i++) {
         userInput[i][0] = sc.nextDouble();
         userInput[i][1] = sc.nextDouble();
      }
      sc.close();
      Matrix evX = new Matrix(userInput);
      Matrix pred = trainer.getMLP().predict(evX);
      pred = pred.apply(v -> (v >= 0.5) ? 1.0 : 0.0);
      System.out.println("Predictions:");
      System.out.println(pred.toIntString());
      // MSE.saveMSE(mse, "mse.csv");
      sc.close();
   }
}
