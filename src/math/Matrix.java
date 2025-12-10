package math;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * A matrix class for mathematical operations on 2D arrays of doubles.
 * Supports various operations including element-wise operations, matrix
 * multiplication,
 * transposition, and functional transformations.
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511052002
 */
public class Matrix {
   private double[][] data;
   private int rows;
   private int cols;

   /**
    * Constructs a matrix with the specified dimensions, initialized with zeros.
    *
    * @param rows the number of rows
    * @param cols the number of columns
    */
   public Matrix(int rows, int cols) {
      if (rows <= 0 || cols <= 0) {
         throw new IllegalArgumentException(
               "Matrix dimensions must be positive.");
      }
      this.data = new double[rows][cols];
      this.rows = rows;
      this.cols = cols;
   }

   /**
    * Constructs a matrix from a 2D array of doubles.
    * Creates a deep copy of the provided data.
    *
    * @param data the 2D array of values to initialize the matrix with
    */
   public Matrix(double[][] data) {
      if (data == null || data.length == 0) {
         throw new IllegalArgumentException(
               "Input array cannot be null or empty.");
      }
      this.rows = data.length;
      this.cols = data[0].length;
      this.data = new double[rows][cols];
      for (int i = 0; i < this.rows; ++i) {
         System.arraycopy(data[i], 0, this.data[i],
               0, this.cols);
      }
   }

   public Matrix(List<double[]> rows) {
      if (rows == null || rows.isEmpty()) {
         throw new IllegalArgumentException(
               "Input list cannot be null or empty.");
      }
      this.rows = rows.size();
      this.cols = rows.get(0).length;
      this.data = new double[this.rows][this.cols];
      for (int i = 0; i < this.rows; ++i) {
         System.arraycopy(rows.get(i), 0, this.data[i],
               0, this.cols);
      }
   }

   /**
    * Copy constructor. Creates a deep copy of another matrix.
    *
    * @param other the matrix to copy
    */
   public Matrix(Matrix other) {
      if (other == null) {
         throw new IllegalArgumentException(
               "Input matrix cannot be null.");
      }
      this.rows = other.rows;
      this.cols = other.cols;
      this.data = new double[rows][cols];
      for (int i = 0; i < this.rows; ++i) {
         System.arraycopy(other.data[i], 0, this.data[i],
               0, this.cols);
      }
   }

   /**
    * Creates a matrix with random values between 0 and 1.
    *
    * @param rows the number of rows
    * @param cols the number of columns
    * @param seed the random seed for reproducibility; if negative, uses current
    *             time
    * @return a new matrix filled with random values
    */
   public static Matrix rand(int rows, int cols, Random rand) {
      Matrix out = new Matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            out.data[i][j] = rand.nextDouble();
         }
      }
      return out;
   }

   /**
    * Creates a matrix with random values using Xavier initialization.
    *
    * @param rows the number of rows (input size)
    * @param cols the number of columns (output size)
    * @param rand Random instance for reproducibility
    * @return a new matrix initialized with Xavier initialization
    */
   public static Matrix randXavier(int rows, int cols, Random rand) {
      Matrix out = new Matrix(rows, cols);
      double limit = Math.sqrt(6.0 / (rows + cols));
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            out.data[i][j] = (rand.nextDouble() * 2 - 1) * limit;
         }
      }
      return out;
   }

   /**
    * Creates a matrix with random values using He initialization.
    * Best for ReLU activation functions.
    * Values are drawn from N(0, sqrt(2/n_in))
    *
    * @param rows the number of rows (input size)
    * @param cols the number of columns (output size)
    * @param rand Random instance for reproducibility
    * @return a new matrix initialized with He initialization
    */
   public static Matrix randHe(int rows, int cols, Random rand) {
      Matrix out = new Matrix(rows, cols);
      double stddev = Math.sqrt(2.0 / rows);
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            out.data[i][j] = rand.nextGaussian() * stddev;
         }
      }
      return out;
   }

   /**
    * Creates a dropout mask matrix with 0s and 1s.
    * Each element is 0 with probability dropoutRate, 1 otherwise.
    * Used for dropout regularization during training.
    *
    * @param rows        the number of rows
    * @param cols        the number of columns
    * @param dropoutRate probability of dropping a neuron (0.0 to 1.0)
    * @param rand        Random instance for reproducibility
    * @return a new dropout mask matrix
    */
   public static Matrix randMask(int rows, int cols, double dropoutRate,
         Random rand) {
      Matrix out = new Matrix(rows, cols);
      double keepProb = 1.0 - dropoutRate;
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            out.data[i][j] = (rand.nextDouble() < keepProb)
                  ? (1.0 / keepProb)
                  : 0.0;
         }
      }
      return out;
   }

   /**
    * Gets the value at the specified position in the matrix.
    *
    * @param row the row index
    * @param col the column index
    * @return the value at the specified position
    */
   public double get(int row, int col) {
      return this.data[row][col];
   }

   /**
    * Returns the number of rows in the matrix.
    *
    * @return the number of rows
    */
   public int rows() {
      return this.rows;
   }

   /**
    * Returns the number of columns in the matrix.
    *
    * @return the number of columns
    */
   public int cols() {
      return this.cols;
   }

   /**
    * Sets this matrix to be a copy of another matrix.
    *
    * @param other the matrix to copy from
    */
   public void set(Matrix other) {
      this.data = other.data;
      this.rows = other.rows;
      this.cols = other.cols;
   }

   /**
    * Extracts specific rows from the matrix using indices from an Array.
    * Creates a new matrix with rows [start, end) from the shuffled indices.
    * Used for mini-batch training.
    *
    * @param indices Array containing row indices (potentially shuffled)
    * @param start   starting index in the indices array (inclusive)
    * @param end     ending index in the indices array (exclusive)
    * @return a new matrix containing the specified rows
    */
   public Matrix rows(Array indices, int start, int end) {
      int batchSize = end - start;
      Matrix result = new Matrix(batchSize, this.cols);
      for (int i = 0; i < batchSize; ++i) {
         System.arraycopy(this.data[indices.get(start + i)], 0,
               result.data[i], 0, this.cols);
      }
      return result;
   }

   // ==============================================================
   // Element operations
   // ==============================================================
   /**
    * Applies a function to each element of the matrix and returns a new matrix.
    *
    * @param fnc the function to apply to each element
    * @return a new matrix with the function applied to all elements
    */
   private Matrix traverse(DoubleUnaryOperator fnc) {
      Matrix result = new Matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            result.data[i][j] = fnc.applyAsDouble(data[i][j]);
         }
      }
      return result;
   }

   /**
    * Applies a function to each element of this matrix in place.
    *
    * @param fnc the function to apply to each element
    */
   private void traverseInPlace(DoubleUnaryOperator fnc) {
      for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
            data[i][j] = fnc.applyAsDouble(data[i][j]);
         }
      }
   }

   /**
    * Applies a function to each element of the matrix and returns a new matrix.
    *
    * @param fnc the function to apply to each element
    * @return a new matrix with the function applied to all elements
    */
   public Matrix apply(DoubleUnaryOperator fnc) {
      return this.traverse(fnc);
   }

   /**
    * Applies a function to each element of this matrix in place.
    *
    * @param fnc the function to apply to each element
    */
   public void applyInPlace(DoubleUnaryOperator fnc) {
      this.traverseInPlace(fnc);
   }

   /**
    * Multiplies each element of the matrix by a scalar and returns a new matrix.
    *
    * @param scalar the scalar value to multiply by
    * @return a new matrix with all elements multiplied by the scalar
    */
   public Matrix mult(double scalar) {
      return this.traverse(x -> x * scalar);
   }

   /**
    * Multiplies each element of this matrix by a scalar in place.
    *
    * @param scalar the scalar value to multiply by
    */
   public void multInPlace(double scalar) {
      this.traverseInPlace(x -> x * scalar);
   }

   /**
    * Adds a scalar to each element of the matrix and returns a new matrix.
    *
    * @param scalar the scalar value to add
    * @return a new matrix with the scalar added to all elements
    */
   public Matrix add(double scalar) {
      return this.traverse(x -> x + scalar);
   }

   /**
    * Adds a scalar to each element of this matrix in place.
    *
    * @param scalar the scalar value to add
    */
   public void addInPlace(double scalar) {
      this.traverseInPlace(x -> x + scalar);
   }

   /**
    * Subtracts a scalar from each element of the matrix and returns a new matrix.
    *
    * @param scalar the scalar value to subtract
    * @return a new matrix with the scalar subtracted from all elements
    */
   public Matrix sub(double scalar) {
      return this.traverse(x -> x - scalar);
   }

   /**
    * Subtracts a scalar from each element of this matrix in place.
    *
    * @param scalar the scalar value to subtract
    */
   public void subInPlace(double scalar) {
      this.traverseInPlace(x -> x - scalar);
   }

   // ==============================================================
   // Element-wise operations between two matrices
   // ==============================================================
   /**
    * Applies a binary function element-wise between this matrix and another
    * matrix.
    *
    * @param other the other matrix
    * @param fnc   the binary function to apply
    * @return a new matrix with the function applied element-wise
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
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

   /**
    * Applies a binary function element-wise between this matrix and another matrix
    * in place.
    *
    * @param other the other matrix
    * @param fnc   the binary function to apply
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
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

   /**
    * Adds another matrix element-wise and returns a new matrix.
    *
    * @param other the matrix to add
    * @return a new matrix with element-wise sum
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public Matrix add(Matrix other) {
      return this.elementWise(other, (a, b) -> a + b);
   }

   /**
    * Adds another matrix element-wise to this matrix in place.
    *
    * @param other the matrix to add
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public void addInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a + b);
   }

   /**
    * Multiplies another matrix element-wise (Hadamard product) and returns a new
    * matrix.
    *
    * @param other the matrix to multiply
    * @return a new matrix with element-wise product
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public Matrix mult(Matrix other) {
      return this.elementWise(other, (a, b) -> a * b);
   }

   /**
    * Multiplies another matrix element-wise (Hadamard product) to this matrix in
    * place.
    *
    * @param other the matrix to multiply
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public void multInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a * b);
   }

   /**
    * Divides this matrix by another matrix element-wise and returns a new matrix.
    *
    * @param other the matrix to divide by
    * @return a new matrix with element-wise division
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public Matrix div(Matrix other) {
      return this.elementWise(other, (a, b) -> a / b);
   }

   /**
    * Divides this matrix by another matrix element-wise in place.
    *
    * @param other the matrix to divide by
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public void divInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a / b);
   }

   /**
    * Subtracts another matrix element-wise and returns a new matrix.
    *
    * @param other the matrix to subtract
    * @return a new matrix with element-wise difference
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public Matrix sub(Matrix other) {
      return this.elementWise(other, (a, b) -> a - b);
   }

   /**
    * Subtracts another matrix element-wise from this matrix in place.
    *
    * @param other the matrix to subtract
    * @throws IllegalArgumentException if matrix dimensions don't match
    */
   public void subInPlace(Matrix other) {
      this.elementWiseInPlace(other, (a, b) -> a - b);
   }

   // ==============================================================
   // Other math operations
   // ==============================================================
   /**
    * Computes the sum of all elements in the matrix.
    *
    * @return the sum of all matrix elements
    */
   public double sum() {
      double total = 0.0;
      for (int i = 0; i < this.rows; ++i) {
         for (int j = 0; j < this.cols; ++j) {
            total += this.data[i][j];
         }
      }
      return total;
   }

   /**
    * Computes the sum of each column and returns a row vector (1 x cols matrix).
    *
    * @return a row vector containing the sum of each column
    */
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

   /**
    * Performs matrix multiplication (dot product) with another matrix.
    *
    * @param other the matrix to multiply with
    * @return a new matrix representing the matrix product
    * @throws IllegalArgumentException if the number of columns in this matrix
    *                                  doesn't match the number of rows in the
    *                                  other matrix
    */
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

   /**
    * Adds a row vector to each row of the matrix and returns a new matrix.
    * This is useful for adding biases in neural networks.
    *
    * @param rowVector a 1 x cols matrix to add to each row
    * @return a new matrix with the row vector added to each row
    * @throws IllegalArgumentException if rowVector is not a row vector or
    *                                  doesn't have the same number of columns
    */
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

   /**
    * Adds a row vector to each row of this matrix in place.
    * This is useful for adding biases in neural networks.
    *
    * @param rowVector a 1 x cols matrix to add to each row
    * @throws IllegalArgumentException if rowVector is not a row vector or
    *                                  doesn't have the same number of columns
    */
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
   /**
    * Returns the transpose of this matrix.
    * Rows and columns are swapped.
    *
    * @return a new matrix that is the transpose of this matrix
    */
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
   /**
    * Compares this matrix with another object for equality.
    * Two matrices are equal if they have the same dimensions and all elements are
    * equal.
    *
    * @param o the object to compare with
    * @return true if the matrices are equal, false otherwise
    */
   @Override
   public boolean equals(Object o) {
      if (!(o instanceof Matrix matrix)) {
         return false;
      }
      return this.rows == matrix.rows && this.cols == matrix.cols &&
            Objects.deepEquals(this.data, matrix.data);
   }

   /**
    * Returns a hash code value for this matrix.
    *
    * @return a hash code based on the matrix data, rows, and columns
    */
   @Override
   public int hashCode() {
      return Objects.hash(Arrays.deepHashCode(this.data), this.rows,
            this.cols);
   }

   // ==============================================================
   // Convert operations
   // ==============================================================
   /**
    * Converts the matrix to a string representation with elements cast to
    * integers.
    * Each row is on a separate line with values separated by spaces.
    *
    * @return a string representation with integer values
    */
   public String toIntString() {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < this.rows; ++i) {
         sb.append((int) this.data[i][0]);
         for (int j = 1; j < this.cols; ++j) {
            sb.append(" ").append((int) this.data[i][j]);
         }
         sb.append("\n");
      }
      return sb.toString();
   }

   /**
    * Returns a string representation of the matrix.
    * Each row is on a separate line with values separated by spaces.
    *
    * @return a string representation of the matrix
    */
   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < this.rows; ++i) {
         sb.append(this.data[i][0]);
         for (int j = 1; j < this.cols; ++j) {
            sb.append(" ").append(this.data[i][j]);
         }
         sb.append("\n");
      }
      return sb.toString();
   }

}
