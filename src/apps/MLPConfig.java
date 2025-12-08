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
   private static final String DATASET_PATH = "data/largeDataset.csv";
   private static final String LABELS_PATH = "data/largeLabels.csv";
   private static final long[] SEEDS = { 42, 97, 123, 456, 789, 1337,
         2023, 9999, 314159, 271828, 123456, 314159, 424242, 8675309 };
   private static final long[] SEEDS1 = { 123, 1337, 9999, 314159 };
   private static final long[] SEEDS2 = { 123 };

   public static void main(String[] args) {
      for (long seed : SEEDS) {
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
      // config1(seed);
      // config2(seed);
      config3(seed);
      config4(seed);
      config5(seed);
      config6(seed);
      config7(seed);
      config8(seed);
      config9(seed);
   }

   private static void config1(long seed) {
      System.out.println("\n=== CONFIG 1 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config1s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            true, false, false,
            true, false, augmentName);
      runTraining(ds, new int[] { 400, 32, 1 },
            0.05, 16000, 800, seed, augmentName);
   }

   private static void config2(long seed) {
      System.out.println("\n=== CONFIG 2 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config2s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, true, false,
            false, false, augmentName);
      runTraining(ds, new int[] { 400, 32, 1 },
            0.03, 16000, 600, seed, augmentName);
   }

   private static void config3(long seed) {
      System.out.println("\n=== CONFIG 3 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config3s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            true, false, false,
            true, false, augmentName);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config4(long seed) {
      System.out.println("\n=== CONFIG 4 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config4s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, true, false,
            false, false, augmentName);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config5(long seed) {
      System.out.println("\n=== CONFIG 5 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config5s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            true, false, false,
            true, false, augmentName);
      runTraining(ds, new int[] { 400, 64, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config6(long seed) {
      System.out.println("\n=== CONFIG 6 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config6s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, true, false,
            false, false, augmentName);
      runTraining(ds, new int[] { 400, 64, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config7(long seed) {
      System.out.println("\n=== CONFIG 7 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config7s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            true, false, false,
            true, false, augmentName);
      runTraining(ds, new int[] { 400, 128, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config8(long seed) {
      System.out.println("\n=== CONFIG 8 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config8s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, true, false,
            false, false, augmentName);
      runTraining(ds, new int[] { 400, 128, 1 },
            0.003, 16000, 600, seed, augmentName);
   }

   private static void config9(long seed) {
      System.out.println("\n=== CONFIG 9 ===");
      StringBuilder augmentName = new StringBuilder("mlp_config9s");
      augmentName.append(seed);
      DataSetBuilder ds = baseDataset(seed, false, false,
            true, false, false,
            true, false, augmentName);
      runTraining(ds, new int[] { 400, 64, 32, 1 },
            0.003, 16000, 600, seed, augmentName);
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
   private static DataSetBuilder baseDataset(long seed, boolean noise,
         boolean elastic, boolean rotation, boolean shift,
         boolean combined1, boolean combined2, boolean combined3,
         StringBuilder augmentName) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> label == 2.0 ? 0.0 : 1.0);
      if (noise) {
         ds.addGaussianNoise(0.005, 1,
               RandomProvider.of(seed));
         augmentName.append("_N");
      }
      if (elastic) {
         ds.addElasticDeformation(8.0, 3.0, 1,
               RandomProvider.of(seed));
         augmentName.append("_E");
      }
      if (rotation) {
         ds.addRotation(10.0, 1,
               RandomProvider.of(seed));
         augmentName.append("_R");
      }
      if (shift) {
         ds.addShift(1, RandomProvider.of(seed));
         augmentName.append("_S");
      }
      if (combined1) {
         ds.addCombinedAugmentation1(1, RandomProvider.of(seed),
               8.0, 3.0, 10.0);
         augmentName.append("_C1");
      }
      if (combined2) {
         ds.addCombinedAugmentation2(1, RandomProvider.of(seed),
               8.0, 3.0, 10.0);
         augmentName.append("_C2");
      }
      if (combined3) {
         ds.addCombinedAugmentation3(1, RandomProvider.of(seed),
               8.0, 3.0, 10.0);
         augmentName.append("_C3");
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
         StringBuilder configName) {
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
      if (evalResult.getAccuracy() < 0.97) {
         System.out.println("WARNING: Low accuracy detected!");
         return;
      }
      try {
         configName.insert(0, "src/ml/models/");
         configName.append(".dat");
         trainer.getMLP().saveModel(configName.toString(),
               evalResult.getOptimalThreshold());
         System.out.println("Model saved: " + configName);
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
