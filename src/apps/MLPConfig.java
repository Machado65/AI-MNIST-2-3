package apps;

import java.util.Arrays;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.EvaluationResult;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import utils.RandomProvider;

/**
 * Optimized MLP configurations for MNIST 2 vs 3 classification.
 * Each configuration varies in architecture, learning rate,
 * number of neurons, and dataset preprocessing.
 * These configurations were determined through systematic experimentation
 * to balance accuracy and training efficiency.
 * Each configuration is encapsulated in its own method for clarity
 * and ease of experimentation.
 */
public class MLPConfig {
   private static final String DATASET_PATH = "data/exLargeDataset.csv";
   private static final String LABELS_PATH = "data/exLargeLabels.csv";
   private static final long[] SEEDS = { 42, 97, 123, 456, 789, 1337,
         2023, 9999, 314159, 271828, 123456, 424242, 8675309 };
   private static final long[] SEEDS1 = { 1337, 314159, 123 };
   private static final long[] SEEDS2 = { 1337 };

   public static void main(String[] args) {
      for (long seed : SEEDS2) {
         System.out.println(
               "=== Running configurations with seed: " + seed + " ===");
         runAllConfigs(seed);
      }
   }

   // ==============================================================
   // OPTIMIZED TRAINING CONFIGURATIONS
   // ==============================================================

   /**
    * Runs all recommended configurations using a given random seed.
    *
    * @param seed Random initialization seed.
    */
   public static void runAllConfigs(long seed) {
      // config1_Baseline(seed);
      // config2_BaselineLowerLr(seed);
      // config3_48Neurons(seed);
      config4_64Neurons(seed);
      // config5_Deeper1(seed);
      // config6_Deeper2(seed);
      config7_Noise(seed);
   }

   private static void config1_Baseline(long seed) {
      System.out.println("\n=== CONFIG 1 ===");
      DataSetBuilder ds = baseDataset(seed, false);
      runTraining(ds, new int[] { 400, 32, 1 },
            0.05, 16000, 800, seed, "mlp_config1s" + seed);
   }

   private static void config2_BaselineLowerLr(long seed) {
      System.out.println("\n=== CONFIG 2 ===");
      DataSetBuilder ds = baseDataset(seed, true);
      runTraining(ds, new int[] { 400, 32, 1 },
            0.05, 16000, 800, seed, "mlp_config2s" + seed);
   }

   private static void config3_48Neurons(long seed) {
      System.out.println("\n=== CONFIG 3 ===");
      DataSetBuilder ds = baseDataset(seed, false);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.01, 16000, 800, seed, "mlp_config3s" + seed);
   }

   private static void config4_64Neurons(long seed) {
      System.out.println("\n=== CONFIG 4 ===");
      DataSetBuilder ds = baseDataset(seed, false);
      runTraining(ds, new int[] { 400, 64, 1 },
            0.008, 14000, 700, seed, "mlp_config4s" + seed);
   }

   private static void config5_Deeper1(long seed) {
      System.out.println("\n=== CONFIG 5 ===");
      DataSetBuilder ds = baseDataset(seed, false);
      runTraining(ds, new int[] { 400, 32, 16, 1 },
            0.008, 14000, 700, seed, "mlp_config5s" + seed);
   }

   private static void config6_Deeper2(long seed) {
      System.out.println("\n=== CONFIG 6 ===");
      DataSetBuilder ds = baseDataset(seed, false);
      runTraining(ds, new int[] { 400, 64, 32, 1 },
            0.005, 16000, 800, seed, "mlp_config6s" + seed);
   }

   private static void config7_Noise(long seed) {
      System.out.println("\n=== CONFIG 7 ===");
      DataSetBuilder ds = baseDataset(seed, true);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.005, 14000, 900, seed, "mlp_config7s" + seed);
   }

   // ==============================================================
   // DATASET PIPELINE
   // ==============================================================

   /**
    * Prepares a dataset with:
    * - Label conversion
    * - Optional normalization
    * - Optional Gaussian noise
    * - Train/test split
    *
    * @param seed      Random seed.
    * @param normalize Whether to normalize features.
    * @param noise     Whether to inject Gaussian noise.
    * @return Configured DataSetBuilder instance.
    */
   private static DataSetBuilder baseDataset(long seed, boolean noise) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> label == 2.0 ? 0.0 : 1.0);
      if (noise) {
         ds.addGaussianNoise(0.005, 1, RandomProvider.of(seed));
      }
      ds.split(0.8, RandomProvider.of(seed));
      return ds;
   }

   // ==============================================================
   // TRAINING PIPELINE
   // ==============================================================

   /**
    * Compiles, trains, evaluates and saves an MLP model.
    *
    * @param ds         Dataset (train/test already prepared)
    * @param topology   Layer topology (including input and output size)
    * @param lr         Learning rate
    * @param epochs     Maximum number of epochs
    * @param patience   Early stopping patience
    * @param seed       Random seed
    * @param configName Name used when saving the model
    */
   private static void runTraining(
         DataSetBuilder ds,
         int[] topology,
         double lr,
         int epochs,
         int patience,
         long seed,
         String configName) {
      System.out.println("Topology     : " + Arrays.toString(topology));
      System.out.println("Learning rate: " + lr);
      System.out.println("Epochs       : " + epochs);
      System.out.println("Patience     : " + patience);
      System.out.println("Seed         : " + seed);
      System.out.println("======================\n");
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      IDifferentiableFunction[] activations = new IDifferentiableFunction[topology.length - 1];
      int n = activations.length - 1;
      for (int i = 0; i < n; ++i) {
         activations[i] = new LeakyReLU();
      }
      // only last activation can be sigmoid as well since we are doing binary
      // classification
      activations[n] = new Sigmoid();
      Trainer trainer = new Trainer(
            topology,
            lr,
            epochs,
            patience,
            activations,
            RandomProvider.of(seed));
      System.out.println(trainer.train(trX, trY, teX, teY));
      EvaluationResult evalResult = trainer.evaluate(teX, teY);
      System.out.println(evalResult);
      if (evalResult.getAccuracy() < 0.95) {
         System.out.println("WARNING: Low accuracy detected!");
         return;
      }
      try {
         trainer.getMLP().saveModel("src/ml/models/" + configName +
               "s" + seed + ".dat");
         System.out.println("Model saved: " + configName);
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
