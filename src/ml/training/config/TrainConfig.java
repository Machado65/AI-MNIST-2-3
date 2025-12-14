package ml.training.config;

import java.util.Random;

/**
 * Configuration class for neural network training.
 * Contains training and testing datasets, hyperparameters, and random seed.
 * Immutable to ensure configuration consistency during training.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class TrainConfig {
   private final DataSet tr;
   private final DataSet te;
   private final double learningRate;
   private final int epochs;
   private final int patience;
   private final Random rand;

   /**
    * Constructs a training configuration with all necessary parameters.
    *
    * @param tr           training dataset (features and labels)
    * @param te           testing/validation dataset (features and labels)
    * @param learningRate learning rate for gradient descent (e.g., 0.01)
    * @param epochs       maximum number of training epochs
    * @param patience     number of epochs without improvement before early
    *                     stopping
    * @param rand         random number generator for reproducibility
    */
   public TrainConfig(DataSet tr, DataSet te, double learningRate,
         int epochs, int patience, Random rand) {
      this.tr = tr;
      this.te = te;
      this.learningRate = learningRate;
      this.epochs = epochs;
      this.patience = patience;
      this.rand = rand;
   }

   /**
    * Copy constructor that creates a deep copy of another configuration.
    *
    * @param other the configuration to copy
    */
   public TrainConfig(TrainConfig other) {
      this.tr = new DataSet(other.tr);
      this.te = new DataSet(other.te);
      this.learningRate = other.learningRate;
      this.epochs = other.epochs;
      this.patience = other.patience;
      this.rand = other.rand;
   }

   /**
    * Returns the training dataset.
    *
    * @return the training dataset
    */
   public DataSet getTr() {
      return this.tr;
   }

   /**
    * Returns the testing/validation dataset.
    *
    * @return the testing dataset
    */
   public DataSet getTe() {
      return this.te;
   }

   /**
    * Returns the learning rate.
    *
    * @return the learning rate
    */
   public double getLearningRate() {
      return this.learningRate;
   }

   /**
    * Returns the maximum number of epochs.
    *
    * @return the maximum epochs
    */
   public int getEpochs() {
      return this.epochs;
   }

   /**
    * Returns the patience for early stopping.
    *
    * @return the patience value
    */
   public int getPatience() {
      return this.patience;
   }

   /**
    * Returns the random number generator.
    *
    * @return the random generator
    */
   public Random getRand() {
      return this.rand;
   }
}
