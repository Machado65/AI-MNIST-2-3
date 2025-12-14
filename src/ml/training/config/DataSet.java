package ml.training.config;

import math.Matrix;

/**
 * Represents a dataset consisting of features (X) and labels (Y).
 * Immutable container for training or testing data.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class DataSet {
   private final Matrix x;
   private final Matrix y;

   /**
    * Constructs a dataset with the given features and labels.
    *
    * @param x the feature matrix where each row is a sample
    * @param y the label matrix where each row is a label
    */
   public DataSet(Matrix x, Matrix y) {
      this.x = x;
      this.y = y;
   }

   /**
    * Copy constructor that creates a deep copy of another dataset.
    *
    * @param other the dataset to copy
    */
   public DataSet(DataSet other) {
      this.x = new Matrix(other.x);
      this.y = new Matrix(other.y);
   }

   /**
    * Returns the feature matrix.
    *
    * @return the feature matrix X
    */
   public Matrix getX() {
      return this.x;
   }

   /**
    * Returns the label matrix.
    *
    * @return the label matrix Y
    */
   public Matrix getY() {
      return this.y;
   }
}
