package ml.training.config;

import java.util.Random;

public class TrainConfig {
   private final DataSet tr;
   private final DataSet te;
   private final double learningRate;
   private final int epochs;
   private final int patience;
   private final Random rand;

   public TrainConfig(DataSet tr, DataSet te, double learningRate,
         int epochs, int patience, Random rand) {
      this.tr = tr;
      this.te = te;
      this.learningRate = learningRate;
      this.epochs = epochs;
      this.patience = patience;
      this.rand = rand;
   }

   public TrainConfig(TrainConfig other) {
      this.tr = new DataSet(other.tr);
      this.te = new DataSet(other.te);
      this.learningRate = other.learningRate;
      this.epochs = other.epochs;
      this.patience = other.patience;
      this.rand = other.rand;
   }

   public DataSet getTr() {
      return this.tr;
   }

   public DataSet getTe() {
      return this.te;
   }

   public double getLearningRate() {
      return this.learningRate;
   }

   public int getEpochs() {
      return this.epochs;
   }

   public int getPatience() {
      return this.patience;
   }

   public Random getRand() {
      return this.rand;
   }
}
