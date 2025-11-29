package ml.training;

import java.util.Random;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;

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
   }
}
