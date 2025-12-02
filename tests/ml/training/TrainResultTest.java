package ml.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("TrainResult Tests")
class TrainResultTest {

   @Nested
   @DisplayName("Constructor and Getter Tests")
   class ConstructorGetterTests {

      @Test
      @DisplayName("Should create TrainResult with all fields")
      void testConstructor() {
         double[] trainMSE = { 0.5, 0.4, 0.3 };
         double[] testMSE = { 0.6, 0.5, 0.4 };
         int bestEpoch = 2;
         double bestTestMSE = 0.4;

         TrainResult result = new TrainResult(trainMSE, testMSE, bestEpoch, bestTestMSE);

         assertNotNull(result);
         assertArrayEquals(trainMSE, result.getTrainMSE());
         assertArrayEquals(testMSE, result.getTestMSE());
         assertEquals(bestEpoch, result.getBestEpoch());
         assertEquals(bestTestMSE, result.getBestTestMSE());
      }

      @Test
      @DisplayName("Should get train MSE array")
      void testGetTrainMSE() {
         double[] trainMSE = { 0.1, 0.2, 0.3 };
         TrainResult result = new TrainResult(trainMSE, new double[3], 0, 0.1);

         double[] retrieved = result.getTrainMSE();
         assertArrayEquals(trainMSE, retrieved);
      }

      @Test
      @DisplayName("Should get test MSE array")
      void testGetTestMSE() {
         double[] testMSE = { 0.15, 0.25, 0.35 };
         TrainResult result = new TrainResult(new double[3], testMSE, 0, 0.15);

         double[] retrieved = result.getTestMSE();
         assertArrayEquals(testMSE, retrieved);
      }

      @Test
      @DisplayName("Should get best epoch")
      void testGetBestEpoch() {
         TrainResult result = new TrainResult(new double[5], new double[5], 3, 0.2);
         assertEquals(3, result.getBestEpoch());
      }

      @Test
      @DisplayName("Should get best test MSE")
      void testGetBestTestMSE() {
         TrainResult result = new TrainResult(new double[5], new double[5], 2, 0.123);
         assertEquals(0.123, result.getBestTestMSE());
      }
   }

   @Nested
   @DisplayName("Edge Cases Tests")
   class EdgeCasesTests {

      @Test
      @DisplayName("Should handle empty MSE arrays")
      void testEmptyArrays() {
         double[] emptyArray = {};
         TrainResult result = new TrainResult(emptyArray, emptyArray, 0, 0.0);

         assertEquals(0, result.getTrainMSE().length);
         assertEquals(0, result.getTestMSE().length);
      }

      @Test
      @DisplayName("Should handle single epoch result")
      void testSingleEpoch() {
         double[] singleMSE = { 0.5 };
         TrainResult result = new TrainResult(singleMSE, singleMSE, 0, 0.5);

         assertEquals(1, result.getTrainMSE().length);
         assertEquals(0, result.getBestEpoch());
         assertEquals(0.5, result.getBestTestMSE());
      }

      @Test
      @DisplayName("Should handle large epoch number")
      void testLargeEpoch() {
         double[] mse = new double[10000];
         TrainResult result = new TrainResult(mse, mse, 9999, 0.001);

         assertEquals(9999, result.getBestEpoch());
      }

      @Test
      @DisplayName("Should handle very small MSE values")
      void testSmallMSE() {
         double[] trainMSE = { 1e-10, 1e-11, 1e-12 };
         double[] testMSE = { 1e-10, 1e-11, 1e-12 };
         TrainResult result = new TrainResult(trainMSE, testMSE, 2, 1e-12);

         assertEquals(1e-12, result.getBestTestMSE());
      }
   }

   @Nested
   @DisplayName("Data Integrity Tests")
   class DataIntegrityTests {

      @Test
      @DisplayName("Should maintain separate arrays for train and test MSE")
      void testSeparateArrays() {
         double[] trainMSE = { 0.1, 0.2, 0.3 };
         double[] testMSE = { 0.15, 0.25, 0.35 };
         TrainResult result = new TrainResult(trainMSE, testMSE, 0, 0.15);

         assertNotSame(result.getTrainMSE(), result.getTestMSE());
      }

      @Test
      @DisplayName("Should have matching array lengths")
      void testMatchingLengths() {
         double[] trainMSE = { 0.1, 0.2, 0.3, 0.4, 0.5 };
         double[] testMSE = { 0.15, 0.25, 0.35, 0.45, 0.55 };
         TrainResult result = new TrainResult(trainMSE, testMSE, 2, 0.35);

         assertEquals(trainMSE.length, result.getTrainMSE().length);
         assertEquals(testMSE.length, result.getTestMSE().length);
      }
   }
}
