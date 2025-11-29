package ml.training;

import java.util.Random;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Step;

public class Trainer {
   private double lr;
   private int epochs;
   private int patience;
   private MLP mlp;

   public Trainer(int[] topology, double lr, int epochs, int patience,
         IDifferentiableFunction[] act, Random rand) {
      this.lr = lr;
      this.epochs = epochs;
      this.patience = patience;
      this.mlp = new MLP(topology, act, rand);
   }

   public TrainResult train(Matrix trX, Matrix trY, Matrix teX, Matrix teY) {
      return mlp.train(trX, trY, teX, teY, lr, epochs, patience);
   }

   public void evaluate(Matrix teX, Matrix teY) {
      // Forward pass
      Matrix pred = mlp.predict(teX);

      // MSE
      Matrix eTe = teY.sub(pred);
      double mse = eTe.dot(eTe.transpose()).get(0, 0) / teX.rows();
      System.out.println("Test MSE: " + mse);

      // Convert to 0/1 classes
      pred = pred.apply(new Step().fnc());
      System.out.println("Predictions (0/1):");
      System.out.println(pred.toIntString());

      // Accuracy
      int correct = 0;
      for (int i = 0; i < teX.rows(); i++) {
         if (pred.get(i, 0) == teY.get(i, 0))
            correct++;
      }
      double accuracy = (double) correct / teX.rows();
      System.out.println("Accuracy: " + accuracy);
   }
}
