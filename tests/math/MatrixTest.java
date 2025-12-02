package math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Matrix Tests")
class MatrixTest {

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should create matrix with specified dimensions")
      void testConstructorWithDimensions() {
         Matrix m = new Matrix(3, 4);
         assertEquals(3, m.rows());
         assertEquals(4, m.cols());
         assertEquals(0.0, m.get(0, 0));
      }

      @Test
      @DisplayName("Should throw exception for non-positive dimensions")
      void testConstructorWithInvalidDimensions() {
         assertThrows(IllegalArgumentException.class, () -> new Matrix(0, 5));
         assertThrows(IllegalArgumentException.class, () -> new Matrix(5, 0));
         assertThrows(IllegalArgumentException.class, () -> new Matrix(-1, 5));
      }

      @Test
      @DisplayName("Should create matrix from 2D array")
      void testConstructorFromArray() {
         double[][] data = { { 1, 2, 3 }, { 4, 5, 6 } };
         Matrix m = new Matrix(data);
         assertEquals(2, m.rows());
         assertEquals(3, m.cols());
         assertEquals(1.0, m.get(0, 0));
         assertEquals(6.0, m.get(1, 2));
      }

      @Test
      @DisplayName("Should throw exception for null or empty array")
      void testConstructorFromInvalidArray() {
         assertThrows(IllegalArgumentException.class, () -> new Matrix((double[][]) null));
         assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[0][]));
      }

      @Test
      @DisplayName("Should create matrix from list of arrays")
      void testConstructorFromList() {
         List<double[]> rows = Arrays.asList(
               new double[] { 1, 2, 3 },
               new double[] { 4, 5, 6 });
         Matrix m = new Matrix(rows);
         assertEquals(2, m.rows());
         assertEquals(3, m.cols());
         assertEquals(1.0, m.get(0, 0));
      }

      @Test
      @DisplayName("Should copy matrix using copy constructor")
      void testCopyConstructor() {
         Matrix original = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix copy = new Matrix(original);
         assertEquals(original.rows(), copy.rows());
         assertEquals(original.cols(), copy.cols());
         assertEquals(original.get(0, 0), copy.get(0, 0));
         assertEquals(original.get(1, 1), copy.get(1, 1));
      }

      @Test
      @DisplayName("Should throw exception for null matrix in copy constructor")
      void testCopyConstructorWithNull() {
         assertThrows(IllegalArgumentException.class, () -> new Matrix((Matrix) null));
      }
   }

   @Nested
   @DisplayName("Static Factory Methods")
   class FactoryMethodTests {

      @Test
      @DisplayName("Should create random matrix with values between 0 and 1")
      void testRand() {
         Random rand = new Random(42);
         Matrix m = Matrix.rand(3, 3, rand);
         assertEquals(3, m.rows());
         assertEquals(3, m.cols());
         for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
               assertTrue(m.get(i, j) >= 0.0 && m.get(i, j) <= 1.0);
            }
         }
      }

      @Test
      @DisplayName("Should create Xavier initialized matrix")
      void testRandXavier() {
         Random rand = new Random(42);
         Matrix m = Matrix.randXavier(3, 3, rand);
         assertEquals(3, m.rows());
         assertEquals(3, m.cols());
         double limit = Math.sqrt(6.0 / 6.0);
         for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
               assertTrue(Math.abs(m.get(i, j)) <= limit);
            }
         }
      }
   }

   @Nested
   @DisplayName("Getter and Setter Tests")
   class GetterSetterTests {

      @Test
      @DisplayName("Should get value at specified position")
      void testGet() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         assertEquals(1.0, m.get(0, 0));
         assertEquals(4.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should return correct dimensions")
      void testRowsAndCols() {
         Matrix m = new Matrix(5, 7);
         assertEquals(5, m.rows());
         assertEquals(7, m.cols());
      }

      @Test
      @DisplayName("Should set matrix from another matrix")
      void testSet() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(2, 2);
         m2.set(m1);
         assertEquals(m1.get(0, 0), m2.get(0, 0));
         assertEquals(m1.get(1, 1), m2.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Element-wise Operations")
   class ElementWiseTests {

      @Test
      @DisplayName("Should apply function to all elements")
      void testApply() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix result = m.apply(x -> x * 2);
         assertEquals(2.0, result.get(0, 0));
         assertEquals(8.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should apply function in place")
      void testApplyInPlace() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         m.applyInPlace(x -> x * 2);
         assertEquals(2.0, m.get(0, 0));
         assertEquals(8.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should multiply by scalar")
      void testMultScalar() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix result = m.mult(2.0);
         assertEquals(2.0, result.get(0, 0));
         assertEquals(8.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should multiply by scalar in place")
      void testMultScalarInPlace() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         m.multInPlace(2.0);
         assertEquals(2.0, m.get(0, 0));
         assertEquals(8.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should add scalar")
      void testAddScalar() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix result = m.add(5.0);
         assertEquals(6.0, result.get(0, 0));
         assertEquals(9.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should add scalar in place")
      void testAddScalarInPlace() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         m.addInPlace(5.0);
         assertEquals(6.0, m.get(0, 0));
         assertEquals(9.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should subtract scalar")
      void testSubScalar() {
         Matrix m = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });
         Matrix result = m.sub(3.0);
         assertEquals(2.0, result.get(0, 0));
         assertEquals(5.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should subtract scalar in place")
      void testSubScalarInPlace() {
         Matrix m = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });
         m.subInPlace(3.0);
         assertEquals(2.0, m.get(0, 0));
         assertEquals(5.0, m.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Matrix-Matrix Operations")
   class MatrixOperationsTests {

      @Test
      @DisplayName("Should add two matrices")
      void testAddMatrix() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });
         Matrix result = m1.add(m2);
         assertEquals(6.0, result.get(0, 0));
         assertEquals(12.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should add two matrices in place")
      void testAddMatrixInPlace() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });
         m1.addInPlace(m2);
         assertEquals(6.0, m1.get(0, 0));
         assertEquals(12.0, m1.get(1, 1));
      }

      @Test
      @DisplayName("Should throw exception when adding incompatible matrices")
      void testAddIncompatibleMatrices() {
         Matrix m1 = new Matrix(2, 3);
         Matrix m2 = new Matrix(3, 2);
         assertThrows(IllegalArgumentException.class, () -> m1.add(m2));
      }

      @Test
      @DisplayName("Should multiply two matrices element-wise")
      void testMultMatrix() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 2, 3 }, { 4, 5 } });
         Matrix result = m1.mult(m2);
         assertEquals(2.0, result.get(0, 0));
         assertEquals(20.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should subtract two matrices")
      void testSubMatrix() {
         Matrix m1 = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });
         Matrix m2 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix result = m1.sub(m2);
         assertEquals(4.0, result.get(0, 0));
         assertEquals(4.0, result.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Matrix Arithmetic")
   class ArithmeticTests {

      @Test
      @DisplayName("Should calculate sum of all elements")
      void testSum() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         assertEquals(10.0, m.sum(), 0.0001);
      }

      @Test
      @DisplayName("Should calculate sum of columns")
      void testSumColumns() {
         Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
         Matrix result = m.sumColumns();
         assertEquals(1, result.rows());
         assertEquals(3, result.cols());
         assertEquals(5.0, result.get(0, 0));
         assertEquals(7.0, result.get(0, 1));
         assertEquals(9.0, result.get(0, 2));
      }

      @Test
      @DisplayName("Should perform matrix multiplication (dot product)")
      void testDot() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 2, 0 }, { 1, 2 } });
         Matrix result = m1.dot(m2);
         assertEquals(4.0, result.get(0, 0));
         assertEquals(4.0, result.get(0, 1));
         assertEquals(10.0, result.get(1, 0));
         assertEquals(8.0, result.get(1, 1));
      }

      @Test
      @DisplayName("Should throw exception for incompatible matrix multiplication")
      void testDotIncompatible() {
         Matrix m1 = new Matrix(2, 3);
         Matrix m2 = new Matrix(2, 2);
         assertThrows(IllegalArgumentException.class, () -> m1.dot(m2));
      }

      @Test
      @DisplayName("Should add row vector to matrix")
      void testAddRowVector() {
         Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
         Matrix rowVec = new Matrix(new double[][] { { 10, 20, 30 } });
         Matrix result = m.addRowVector(rowVec);
         assertEquals(11.0, result.get(0, 0));
         assertEquals(22.0, result.get(0, 1));
         assertEquals(14.0, result.get(1, 0));
      }

      @Test
      @DisplayName("Should add row vector in place")
      void testAddRowVectorInPlace() {
         Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
         Matrix rowVec = new Matrix(new double[][] { { 10, 20, 30 } });
         m.addInPlaceRowVector(rowVec);
         assertEquals(11.0, m.get(0, 0));
         assertEquals(22.0, m.get(0, 1));
      }

      @Test
      @DisplayName("Should throw exception for invalid row vector")
      void testAddRowVectorInvalid() {
         Matrix m = new Matrix(2, 3);
         Matrix notRowVec = new Matrix(2, 2);
         assertThrows(IllegalArgumentException.class, () -> m.addRowVector(notRowVec));
      }
   }

   @Nested
   @DisplayName("Transpose and Transformations")
   class TransformationTests {

      @Test
      @DisplayName("Should transpose matrix")
      void testTranspose() {
         Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
         Matrix t = m.transpose();
         assertEquals(3, t.rows());
         assertEquals(2, t.cols());
         assertEquals(1.0, t.get(0, 0));
         assertEquals(6.0, t.get(2, 1));
         assertEquals(4.0, t.get(0, 1));
      }

      @Test
      @DisplayName("Should transpose square matrix")
      void testTransposeSquare() {
         Matrix m = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix t = m.transpose();
         assertEquals(1.0, t.get(0, 0));
         assertEquals(3.0, t.get(0, 1));
         assertEquals(2.0, t.get(1, 0));
         assertEquals(4.0, t.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Comparison and Equality")
   class EqualityTests {

      @Test
      @DisplayName("Should be equal to identical matrix")
      void testEquals() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         assertEquals(m1, m2);
      }

      @Test
      @DisplayName("Should not be equal to different matrix")
      void testNotEquals() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 1, 2 }, { 3, 5 } });
         assertNotEquals(m1, m2);
      }

      @Test
      @DisplayName("Should not be equal to null")
      void testNotEqualsNull() {
         Matrix m = new Matrix(2, 2);
         assertNotEquals(null, m);
      }

      @Test
      @DisplayName("Should have consistent hashCode")
      void testHashCode() {
         Matrix m1 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         Matrix m2 = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
         assertEquals(m1.hashCode(), m2.hashCode());
      }
   }

   @Nested
   @DisplayName("String Conversion")
   class StringConversionTests {

      @Test
      @DisplayName("Should convert to string")
      void testToString() {
         Matrix m = new Matrix(new double[][] { { 1.5, 2.5 }, { 3.5, 4.5 } });
         String str = m.toString();
         assertNotNull(str);
         assertTrue(str.contains("1.5"));
         assertTrue(str.contains("4.5"));
      }

      @Test
      @DisplayName("Should convert to int string")
      void testToIntString() {
         Matrix m = new Matrix(new double[][] { { 1.7, 2.3 }, { 3.9, 4.1 } });
         String str = m.toIntString();
         assertNotNull(str);
         assertTrue(str.contains("1"));
         assertTrue(str.contains("4"));
      }
   }
}
