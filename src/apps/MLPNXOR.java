package apps;

import java.util.Scanner;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.activation.Step;

/**
 * Application that trains a Multi-Layer Perceptron (MLP) to learn the XOR
 * function.
 * The XOR problem is a classic non-linearly separable problem that requires at
 * least
 * one hidden layer to solve.
 *
 * This program:
 * - Trains an MLP with 2 input neurons, 2 hidden neurons, and 1 output neuron
 * - Uses sigmoid activation functions for hidden and output layers
 * - Reads test inputs from standard input
 * - Outputs predictions, learned weights, biases, and saves MSE history
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511050300
 */
public class MLPNXOR {
   public static void main(String[] args) {
      double lr = 0.1;
      int epochs = 50000;
      int[] topology = { 2, 2, 1 };
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
      // Get input and create evaluation Matrix
      Scanner sc = new Scanner(System.in);
      int n = sc.nextInt();
      double[][] input = new double[n][2];
      for (int i = 0; i < n; ++i) {
         input[i][0] = sc.nextDouble();
         input[i][1] = sc.nextDouble();
      }
      Matrix evX = new Matrix(input);
      // Train the MLP
      MLP mlp = new MLP(topology,
            new IDifferentiableFunction[] {
                  new Sigmoid(),
                  new Sigmoid(), },
            65);
      double[] mse = mlp.train(trX, trY, lr, epochs);
      // Predict and output results
      Matrix pred = mlp.predict(evX);
      // System.out.println("Weights:");
      // int layer = 1;
      // for (Matrix w : mlp.getWeights()) {
      // System.out.println("Layer " + layer + ":");
      // System.out.print(w);
      // layer++;
      // }
      // layer = 1;
      // System.out.println("Biases:");
      // for (Matrix b : mlp.getBiases()) {
      // System.out.println("Layer " + layer + ":");
      // System.out.print(b);
      // layer++;
      // }
      // convert probabilities to integer classes: 0 or 1
      pred = pred.apply(new Step().fnc());
      // print output
      // insert code here to print the pred Matrix as integers
      // …
      // System.out.println("Predictions:");
      System.out.print(pred.toIntString());
      // MSE.saveMSE(mse, "mse.csv");
      sc.close();
   }
}
