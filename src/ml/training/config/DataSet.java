package ml.training.config;

import math.Matrix;

public class DataSet {
   private final Matrix x;
   private final Matrix y;

   public DataSet(Matrix x, Matrix y) {
      this.x = x;
      this.y = y;
   }

   public DataSet(DataSet other) {
      this.x = new Matrix(other.x);
      this.y = new Matrix(other.y);
   }

   public Matrix getX() {
      return this.x;
   }

   public Matrix getY() {
      return this.y;
   }
}
