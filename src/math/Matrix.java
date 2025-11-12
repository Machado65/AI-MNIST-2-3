package math;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511052002
 */
public class Matrix {
   private double[][] data;
   private int rows;
   private int cols;

   public Matrix(int rows, int cols) {
      this.data = new double[rows][cols];
      this.rows = rows;
      this.cols = cols;
   }

   public Matrix(double[][] data) {
      this.rows = data.length;
      this.cols = data[0].length;
      this.data = new double[rows][cols];
      for (int i = 0; i < this.rows; ++i) {
         System.arraycopy(data[i], 0, this.data[i], 0,
               this.cols);
      }
   }

   public Matrix(Matrix other) {
      this.rows = other.rows;
      this.cols = other.cols;
      this.data = new double[rows][cols];
      for (int i = 0; i < this.rows; ++i) {
         System.arraycopy(other.data[i], 0, this.data[i],
               0, this.cols);
      }
   }

   public static Matrix rand(int rows, int cols, int seed) {
      Matrix out = new Matrix(rows, cols);
      if (seed < 0) {
         seed = (int) System.currentTimeMillis();
      }
      Random rand = new Random(seed);
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            out.data[i][j] = rand.nextDouble();
         }
      }
      return out;
   }

   public double get(int row, int col) {
      return this.data[row][col];
   }

   public int rows() {
      return this.rows;
   }

   public int cols() {
      return this.cols;
   }

   public void set(Matrix other) {
      this.data = other.data;
      this.rows = other.rows;
      this.cols = other.cols;
   }

   // ==============================================================
   // Element operations
   // ==============================================================
   private Matrix traverse(DoubleUnaryOperator fnc) {
      Matrix result = new Matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            result.data[i][j] = fnc.applyAsDouble(data[i][j]);
         }
      }
      return result;
   }

   private void traverseInPlace(DoubleUnaryOperator fnc) {
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            data[i][j] = fnc.applyAsDouble(data[i][j]);
         }
      }
   }

   public Matrix apply(DoubleUnaryOperator fnc) {
      return this.traverse(fnc);
   }

   public void applyInPlace(DoubleUnaryOperator fnc) {
      this.traverseInPlace(fnc);
   }

   public Matrix mult(double scalar) {
      return this.traverse(x -> x * scalar);
   }

   public void multInPlace(double scalar) {
      this.traverseInPlace(x -> x * scalar);
   }

   public Matrix add(double scalar) {
      return this.traverse(x -> x + scalar);
   }

   public void addInPlace(double scalar) {
      this.traverseInPlace(x -> x + scalar);
   }

   public Matrix sub(double scalar) {
      return this.traverse(x -> x - scalar);
   }

   public void subInPlace(double scalar) {
      this.traverseInPlace(x -> x - scalar);
   }

   // ==============================================================
   // Element-wise operations between two matrices
   // ==============================================================
   private Matrix elementWise(Matrix other, DoubleBinaryOperator fnc) {
      if (this.rows != other.rows || this.cols != other.cols) {
         throw new IllegalArgumentException(
               "Incompatible matrix sizes for element wise.");
      }
      Matrix result = new Matrix(this.rows, this.cols);
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j)
            result.data[i][j] = fnc.applyAsDouble(this.data[i][j],
                  other.data[i][j]);
      }
      return result;
   }

   private void elementWiseInPlace(Matrix other,
         DoubleBinaryOperator fnc) {
      if (this.rows != other.rows || this.cols != other.cols) {
         throw new IllegalArgumentException(
               "Incompatible matrix sizes for element wise.");
      }
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j)
            this.data[i][j] = fnc.applyAsDouble(this.data[i][j],
                  other.data[i][j]);
      }
   }

   public Matrix add(Matrix other) {
      return this.elementWise(other, (a, b) -> a + b);
   }

   public void addInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a + b);
   }

   public Matrix mult(Matrix other) {
      return this.elementWise(other, (a, b) -> a * b);
   }

   public void multInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a * b);
   }

   public Matrix sub(Matrix other) {
      return this.elementWise(other, (a, b) -> a - b);
   }

   public void subInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a - b);
   }

   // ==============================================================
   // Other math operations
   // ==============================================================
   public double sum() {
      double total = 0.0;
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            total += this.data[i][j];
         }
      }
      return total;
   }

   public Matrix sumColumns() {
      Matrix result = new Matrix(1, this.cols);
      for (int j = 0; j < this.cols; ++j) {
         double sum = 0.0;
         for (int i = 0; i < this.rows; ++i) {
            sum += this.data[i][j];
         }
         result.data[0][j] = sum;
      }
      return result;
   }

   public Matrix dot(Matrix other) {
      if (this.cols != other.rows) {
         throw new IllegalArgumentException(
               "Incompatible matrix sizes for multiplication.");
      }
      Matrix result = new Matrix(this.rows, other.cols);
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < this.cols; ++k) {
               result.data[i][j] += this.data[i][k] * other.data[k][j];
            }
         }
      }
      return result;
   }

   public Matrix addRowVector(Matrix rowVector) {
      if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {
         throw new IllegalArgumentException(
               "Incompatible sizes for adding row vector.");
      }
      Matrix result = new Matrix(this.rows, this.cols);
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            result.data[i][j] = this.data[i][j] + rowVector.data[0][j];
         }
      }
      return result;
   }

   public void addInPlaceRowVector(Matrix rowVector) {
      if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {
         throw new IllegalArgumentException(
               "Incompatible sizes for adding row vector.");
      }
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            this.data[i][j] += rowVector.data[0][j];
         }
      }
   }

   // ==============================================================
   // Column and row operations
   // ==============================================================
   public Matrix transpose() {
      Matrix result = new Matrix(cols, rows);
      // transpose the matrix
      // store the result in matrix result
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
         }
      }
      return result;
   }

   // ==============================================================
   // Compare operations
   // ==============================================================
   @Override
   public boolean equals(Object o) {
      if (!(o instanceof Matrix matrix)) {
         return false;
      }
      return this.rows == matrix.rows && this.cols == matrix.cols &&
            Objects.deepEquals(this.data, matrix.data);
   }

   @Override
   public int hashCode() {
      return Objects.hash(Arrays.deepHashCode(this.data), this.rows,
            this.cols);
   }

   // ==============================================================
   // Convert operations
   // ==============================================================
   public String toIntString() {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            sb.append((int) this.data[i][j]).append(" ");
         }
         sb.append("\n");
      }
      return sb.toString();
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            sb.append(this.data[i][j]).append(" ");
         }
         sb.append("\n");
      }
      return sb.toString();
   }
}
