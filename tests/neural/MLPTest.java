package neural;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import math.Matrix;
import ml.training.TrainResult;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;

@DisplayName("MLP (Multi-Layer Perceptron) Tests")
class MLPTest {

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should create MLP with valid architecture")
      void testConstructor() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         Random rand = new Random(42);

         MLP mlp = new MLP(topology, act, rand);

         assertNotNull(mlp);
         assertNotNull(mlp.getWeights());
         assertNotNull(mlp.getBiases());
         assertEquals(2, mlp.getWeights().length);
         assertEquals(2, mlp.getBiases().length);
      }

      @Test
      @DisplayName("Should create MLP with multiple hidden layers")
      void testMultipleHiddenLayers() {
         int[] topology = { 10, 8, 6, 4, 2 };
         IDifferentiableFunction[] act = {
               new Sigmoid(), new Sigmoid(), new Sigmoid(), new Sigmoid()
         };
         Random rand = new Random(42);

         MLP mlp = new MLP(topology, act, rand);

         assertEquals(4, mlp.getWeights().length);
         assertEquals(4, mlp.getBiases().length);
      }

      @Test
      @DisplayName("Should initialize weights with correct dimensions")
      void testWeightDimensions() {
         int[] topology = { 3, 4, 2 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         Random rand = new Random(42);

         MLP mlp = new MLP(topology, act, rand);
         Matrix[] weights = mlp.getWeights();

         assertEquals(3, weights[0].rows());
         assertEquals(4, weights[0].cols());
         assertEquals(4, weights[1].rows());
         assertEquals(2, weights[1].cols());
      }

      @Test
      @DisplayName("Should initialize biases with correct dimensions")
      void testBiasDimensions() {
         int[] topology = { 3, 4, 2 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         Random rand = new Random(42);

         MLP mlp = new MLP(topology, act, rand);
         Matrix[] biases = mlp.getBiases();

         assertEquals(1, biases[0].rows());
         assertEquals(4, biases[0].cols());
         assertEquals(1, biases[1].rows());
         assertEquals(2, biases[1].cols());
      }
   }

   @Nested
   @DisplayName("Prediction Tests")
   class PredictionTests {

      @Test
      @DisplayName("Should predict with correct output dimensions")
      void testPredictDimensions() {
         int[] topology = { 3, 4, 2 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix input = new Matrix(new double[][] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } });
         Matrix output = mlp.predict(input);

         assertEquals(2, output.rows());
         assertEquals(2, output.cols());
      }

      @Test
      @DisplayName("Should predict single sample")
      void testPredictSingleSample() {
         int[] topology = { 2, 2, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix input = new Matrix(new double[][] { { 0.5, 0.5 } });
         Matrix output = mlp.predict(input);

         assertEquals(1, output.rows());
         assertEquals(1, output.cols());
         assertTrue(output.get(0, 0) >= 0.0 && output.get(0, 0) <= 1.0);
      }

      @Test
      @DisplayName("Should predict batch of samples")
      void testPredictBatch() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix input = new Matrix(new double[][] {
               { 0.0, 0.0 },
               { 0.0, 1.0 },
               { 1.0, 0.0 },
               { 1.0, 1.0 }
         });
         Matrix output = mlp.predict(input);

         assertEquals(4, output.rows());
         assertEquals(1, output.cols());
      }
   }

   @Nested
   @DisplayName("Training Tests")
   class TrainingTests {

      @Test
      @DisplayName("Should train and return MSE array")
      void testTrainReturnsMSE() {
         int[] topology = { 2, 2, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix x = new Matrix(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
         Matrix y = new Matrix(new double[][] { { 0 }, { 1 }, { 1 }, { 0 } });

         double[] mse = mlp.train(x, y, 0.05, 500);

         assertEquals(500, mse.length);
         assertTrue(mse[mse.length - 1] < mse[0] + 1e-6);
      }

      @Test
      @DisplayName("Should train with early stopping")
      void testTrainWithEarlyStopping() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 } });
         Matrix trY = new Matrix(new double[][] { { 0 }, { 1 }, { 1 } });
         Matrix teX = new Matrix(new double[][] { { 1, 1 } });
         Matrix teY = new Matrix(new double[][] { { 0 } });

         TrainResult result = mlp.train(trX, trY, teX, teY, 0.1, 1000, 100);

         assertNotNull(result);
         assertTrue(result.getBestEpoch() < 1000);
         assertTrue(result.getBestTestMSE() >= 0.0);
      }

      @Test
      @DisplayName("Should reduce training error over epochs")
      void testTrainingReducesError() {
         int[] topology = { 2, 4, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix x = new Matrix(new double[][] { { 0, 0 }, { 1, 1 } });
         Matrix y = new Matrix(new double[][] { { 0 }, { 1 } });

         double[] mse = mlp.train(x, y, 0.5, 500);

         assertTrue(mse[499] < mse[0]);
      }
   }

   @Nested
   @DisplayName("Weight and Bias Tests")
   class WeightBiasTests {

      @Test
      @DisplayName("Should get weights reference")
      void testGetWeights() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix[] weights = mlp.getWeights();
         assertNotNull(weights);
         assertEquals(2, weights.length);
      }

      @Test
      @DisplayName("Should get weights copy")
      void testGetWeightsCopy() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix[] weightsCopy = mlp.getWeightsCopy();
         Matrix[] weightsRef = mlp.getWeights();

         assertNotNull(weightsCopy);
         assertEquals(weightsRef.length, weightsCopy.length);
         // Verify it's a copy, not the same reference
         for (int i = 0; i < weightsCopy.length; i++) {
            assertNotSame(weightsRef[i], weightsCopy[i]);
         }
      }

      @Test
      @DisplayName("Should get biases reference")
      void testGetBiases() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix[] biases = mlp.getBiases();
         assertNotNull(biases);
         assertEquals(2, biases.length);
      }

      @Test
      @DisplayName("Should get biases copy")
      void testGetBiasesCopy() {
         int[] topology = { 2, 3, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix[] biasesCopy = mlp.getBiasesCopy();
         Matrix[] biasesRef = mlp.getBiases();

         assertNotNull(biasesCopy);
         assertEquals(biasesRef.length, biasesCopy.length);
         // Verify it's a copy, not the same reference
         for (int i = 0; i < biasesCopy.length; i++) {
            assertNotSame(biasesRef[i], biasesCopy[i]);
         }
      }
   }

   @Nested
   @DisplayName("XOR Problem Tests")
   class XORProblemTests {

      @Test
      @DisplayName("Should learn XOR function")
      void testLearnXOR() {
         int[] topology = { 2, 4, 1 };
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         MLP mlp = new MLP(topology, act, new Random(42));

         Matrix x = new Matrix(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
         Matrix y = new Matrix(new double[][] { { 0 }, { 1 }, { 1 }, { 0 } });

         mlp.train(x, y, 0.5, 5000);
         Matrix predictions = mlp.predict(x);

         // Check predictions are reasonable (within 0.3 of targets)
         assertTrue(Math.abs(predictions.get(0, 0) - 0.0) < 0.3);
         assertTrue(Math.abs(predictions.get(1, 0) - 1.0) < 0.3);
         assertTrue(Math.abs(predictions.get(2, 0) - 1.0) < 0.3);
         assertTrue(Math.abs(predictions.get(3, 0) - 0.0) < 0.3);
      }
   }
}
