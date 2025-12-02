package ml.training;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;

@DisplayName("Trainer Tests")
class TrainerTest {

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should create Trainer with valid parameters")
      void testConstructor() {
         int[] topology = { 2, 3, 1 };
         double lr = 0.01;
         int epochs = 1000;
         int patience = 100;
         IDifferentiableFunction[] act = { new Sigmoid(), new Sigmoid() };
         Random rand = new Random(42);

         Trainer trainer = new Trainer(topology, lr, epochs, patience, act, rand);

         assertNotNull(trainer);
      }

      @Test
      @DisplayName("Should create Trainer with small learning rate")
      void testSmallLearningRate() {
         int[] topology = { 10, 5, 1 };
         Trainer trainer = new Trainer(
               topology, 0.001, 100, 10,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));
         assertNotNull(trainer);
      }

      @Test
      @DisplayName("Should create Trainer with large network")
      void testLargeNetwork() {
         int[] topology = { 100, 64, 32, 16, 1 };
         IDifferentiableFunction[] act = {
               new Sigmoid(), new Sigmoid(), new Sigmoid(), new Sigmoid()
         };
         Trainer trainer = new Trainer(topology, 0.01, 1000, 100, act, new Random(42));
         assertNotNull(trainer);
      }
   }

   @Nested
   @DisplayName("Training Tests")
   class TrainingTests {

      @Test
      @DisplayName("Should train and return TrainResult")
      void testTrain() {
         int[] topology = { 2, 3, 1 };
         Trainer trainer = new Trainer(
               topology, 0.1, 100, 10,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
         Matrix trY = new Matrix(new double[][] { { 0 }, { 1 }, { 1 }, { 0 } });
         Matrix teX = new Matrix(new double[][] { { 0, 0 }, { 1, 1 } });
         Matrix teY = new Matrix(new double[][] { { 0 }, { 0 } });

         TrainResult result = trainer.train(trX, trY, teX, teY);

         assertNotNull(result);
         assertNotNull(result.getTrainMSE());
         assertNotNull(result.getTestMSE());
         assertTrue(result.getBestEpoch() >= 0);
         assertTrue(result.getBestTestMSE() >= 0.0);
      }

      @Test
      @DisplayName("Should train with separate train and test sets")
      void testTrainSeparateSets() {
         int[] topology = { 3, 5, 2 };
         Trainer trainer = new Trainer(
               topology, 0.05, 200, 20,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } });
         Matrix trY = new Matrix(new double[][] { { 1, 0 }, { 0, 1 } });
         Matrix teX = new Matrix(new double[][] { { 0.2, 0.3, 0.4 } });
         Matrix teY = new Matrix(new double[][] { { 1, 0 } });

         TrainResult result = trainer.train(trX, trY, teX, teY);

         assertNotNull(result);
         assertTrue(result.getTrainMSE().length > 0);
         assertTrue(result.getTestMSE().length > 0);
      }

      @Test
      @DisplayName("Should find best epoch during training")
      void testFindsBestEpoch() {
         int[] topology = { 2, 4, 1 };
         Trainer trainer = new Trainer(
               topology, 0.1, 500, 50,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 } });
         Matrix trY = new Matrix(new double[][] { { 0 }, { 1 }, { 1 } });
         Matrix teX = new Matrix(new double[][] { { 1, 1 } });
         Matrix teY = new Matrix(new double[][] { { 0 } });

         TrainResult result = trainer.train(trX, trY, teX, teY);

         int bestEpoch = result.getBestEpoch();
         assertTrue(bestEpoch >= 0 && bestEpoch < 500);
      }
   }

   @Nested
   @DisplayName("Evaluation Tests")
   class EvaluationTests {

      @Test
      @DisplayName("Should evaluate model on test set")
      void testEvaluate() {
         int[] topology = { 2, 3, 1 };
         Trainer trainer = new Trainer(
               topology, 0.1, 100, 10,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0, 0 }, { 1, 1 } });
         Matrix trY = new Matrix(new double[][] { { 0 }, { 1 } });
         Matrix teX = new Matrix(new double[][] { { 0, 1 } });
         Matrix teY = new Matrix(new double[][] { { 1 } });

         trainer.train(trX, trY, teX, teY);

         // evaluate method doesn't return anything, just testing it doesn't throw
         assertDoesNotThrow(() -> trainer.evaluate(teX, teY));
      }

      @Test
      @DisplayName("Should evaluate after training")
      void testEvaluateAfterTraining() {
         int[] topology = { 3, 4, 2 };
         Trainer trainer = new Trainer(
               topology, 0.01, 50, 10,
               new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
               new Random(42));

         Matrix trX = new Matrix(new double[][] { { 0.1, 0.2, 0.3 } });
         Matrix trY = new Matrix(new double[][] { { 1, 0 } });
         Matrix teX = new Matrix(new double[][] { { 0.4, 0.5, 0.6 } });
         Matrix teY = new Matrix(new double[][] { { 0, 1 } });

         trainer.train(trX, trY, teX, teY);
         trainer.evaluate(teX, teY);

         // If we reach here, evaluation completed without errors
         assertTrue(true);
      }
   }

   @Nested
   @DisplayName("Hyperparameter Tests")
   class HyperparameterTests {

      @Test
      @DisplayName("Should handle different learning rates")
      void testDifferentLearningRates() {
         double[] learningRates = { 0.001, 0.01, 0.1, 0.5 };

         for (double lr : learningRates) {
            Trainer trainer = new Trainer(
                  new int[] { 2, 3, 1 }, lr, 50, 10,
                  new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
                  new Random(42));

            Matrix trX = new Matrix(new double[][] { { 0, 0 } });
            Matrix trY = new Matrix(new double[][] { { 0 } });

            TrainResult result = trainer.train(trX, trY, trX, trY);
            assertNotNull(result);
         }
      }

      @Test
      @DisplayName("Should handle different patience values")
      void testDifferentPatience() {
         int[] patienceValues = { 1, 10, 100, 1000 };

         for (int patience : patienceValues) {
            Trainer trainer = new Trainer(
                  new int[] { 2, 2, 1 }, 0.1, 100, patience,
                  new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
                  new Random(42));

            Matrix trX = new Matrix(new double[][] { { 0, 0 }, { 1, 1 } });
            Matrix trY = new Matrix(new double[][] { { 0 }, { 1 } });

            TrainResult result = trainer.train(trX, trY, trX, trY);
            assertNotNull(result);
         }
      }

      @Test
      @DisplayName("Should handle different epoch counts")
      void testDifferentEpochs() {
         int[] epochCounts = { 10, 100, 1000 };

         for (int epochs : epochCounts) {
            Trainer trainer = new Trainer(
                  new int[] { 2, 2, 1 }, 0.1, epochs, 50,
                  new IDifferentiableFunction[] { new Sigmoid(), new Sigmoid() },
                  new Random(42));

            Matrix trX = new Matrix(new double[][] { { 0, 0 } });
            Matrix trY = new Matrix(new double[][] { { 0 } });

            TrainResult result = trainer.train(trX, trY, trX, trY);
            assertNotNull(result);
            assertTrue(result.getBestEpoch() < epochs);
         }
      }
   }
}
