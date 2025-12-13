package apps;

import java.util.Arrays;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.config.DataSet;
import ml.training.config.TrainConfig;
import ml.training.core.Trainer;
import ml.training.result.EvaluationResult;
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
   private static final long[] SEEDS1 = { 42, 1337, 2024, 9999 };
   private static final long[] SEEDS2 = { 1337, 2023, 9999, 314159 };
   private static final long[] SEEDS3 = { 2023 };
   private static final double[] stDevs = { 0.01, 0.02, 0.03 };
   private static final double[] alphas = { 4.0, 5.0, 6.0 };
   private static final double[] sigmas = { 1.5, 2.0, 2.5 };
   private static final double[] rots = { 3.0, 5.0, 7.0 };
   private static final int[] shifts = { 1, 2 };
   private static final double stDev = 0.02;
   private static final double alpha = 6.0;
   private static final double sigma = 2.0;
   private static final double maxDegrees = 5.0;
   private static final int shiftPixels = 1;

   public static void main(String[] args) {
      for (long seed : SEEDS) {
         System.out.println(
               "=== Running configurations with seed: " + seed + " ===");
         runHyperparameterSearch(seed);
      }
   }

   public static void runHyperparameterSearch(long seed) {
      runAllConfigs(seed, stDev, alpha, sigma, maxDegrees, shiftPixels);
   }

   // ==============================================================
   // OPTIMIZED TRAINING CONFIGURATIONS
   // ==============================================================

   /**
    * Runs all recommended configurations using a given random seed.
    *
    * @param seed Random initialization seed.
    */
   public static void runAllConfigs(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      // config0(seed);
      config1(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      config2(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config3(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config4(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config5(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config6(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config7(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config8(seed, stdDev, alpha, sigma, maxDegrees, shiftPixels);
      // config9(seed);
   }

   private static void config0(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 0 ===");
      StringBuilder augmentName = new StringBuilder("config0");
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, false, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 512, 1 },
            0.002, 16000, 1200, seed, augmentName);
   }

   private static void config1(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 1 ===");
      StringBuilder augmentName = new StringBuilder("config1");
      DataSetBuilder ds = baseDataset(seed, true, true,
            true, false, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 256, 1 },
            0.002, 16000, 1200, seed, augmentName);
   }

   private static void config2(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 2 ===");
      StringBuilder augmentName = new StringBuilder("config2");
      DataSetBuilder ds = baseDataset(seed, false, true,
            true, true, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 256, 1 }, 0.002, 16000,
            1200, seed, augmentName);
   }

   private static void config3(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 3 ===");
      StringBuilder augmentName = new StringBuilder("config3");
      DataSetBuilder ds = baseDataset(seed, true, true,
            false, false, false,
            true, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.002, 16000, 800, seed, augmentName);
   }

   private static void config4(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 4 ===");
      StringBuilder augmentName = new StringBuilder("config4");
      DataSetBuilder ds = baseDataset(seed, false, true,
            true, false, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 48, 1 },
            0.002, 16000, 800, seed, augmentName);
   }

   private static void config5(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 5 ===");
      StringBuilder augmentName = new StringBuilder("config5");
      DataSetBuilder ds = baseDataset(seed, true, true,
            false, false, false,
            true, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 64, 1 },
            0.002, 16000, 800, seed, augmentName);
   }

   private static void config6(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 6 ===");
      StringBuilder augmentName = new StringBuilder("config6");
      DataSetBuilder ds = baseDataset(seed, false, true,
            true, false, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 64, 1 },
            0.002, 16000, 800, seed, augmentName);
   }

   private static void config7(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 7 ===");
      StringBuilder augmentName = new StringBuilder("config7");
      DataSetBuilder ds = baseDataset(seed, true, true,
            false, false, false,
            true, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 128, 1 },
            0.002, 16000, 1000, seed, augmentName);
   }

   private static void config8(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 8 ===");
      StringBuilder augmentName = new StringBuilder("config8");
      DataSetBuilder ds = baseDataset(seed, false, true,
            true, false, false,
            false, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 128, 1 },
            0.002, 16000, 1000, seed, augmentName);
   }

   private static void config9(long seed, double stdDev, double alpha,
         double sigma, double maxDegrees, int shiftPixels) {
      System.out.println("\n=== CONFIG 9 ===");
      StringBuilder augmentName = new StringBuilder("config9");
      DataSetBuilder ds = baseDataset(seed, false, false,
            false, false, false,
            true, false, stdDev, alpha, sigma,
            maxDegrees, shiftPixels, augmentName);
      runTraining(ds, new int[] { 400, 64, 32, 1 },
            0.002, 16000, 800, seed, augmentName);
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
         double stdDev, double alpha, double sigma, double maxDegrees,
         int shiftPixels, StringBuilder augmentName) {
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> label == 2.0 ? 0.0 : 1.0);
      if (noise) {
         ds.addGaussianNoise(stdDev, 1,
               RandomProvider.of(seed));
         augmentName.append("_N:").append(stdDev);
      }
      if (elastic) {
         ds.addElasticDeformation(alpha, sigma, 1,
               RandomProvider.of(seed));
         augmentName.append("_E:").append(alpha).append("_")
               .append(sigma);
      }
      if (rotation) {
         ds.addRotation(maxDegrees, 1,
               RandomProvider.of(seed));
         augmentName.append("_R:").append(maxDegrees);
      }
      if (shift) {
         ds.addShift(shiftPixels, 1, RandomProvider.of(seed));
         augmentName.append("_S:").append(shiftPixels);
      }
      if (combined1) {
         ds.addCombinedAugmentation1(1, RandomProvider.of(seed),
               alpha, sigma, maxDegrees);
         augmentName.append("_C1").append(":")
               .append(alpha).append("_")
               .append(sigma).append("_")
               .append(maxDegrees);
      }
      if (combined2) {
         ds.addCombinedAugmentation2(1, RandomProvider.of(seed),
               maxDegrees, shiftPixels);
         augmentName.append("_C2").append(":")
               .append(maxDegrees).append("_")
               .append(shiftPixels);
      }
      if (combined3) {
         ds.addCombinedAugmentation3(1, RandomProvider.of(seed),
               stdDev, sigma);
         augmentName.append("_C3").append(":")
               .append(stdDev).append("_")
               .append(sigma);
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
   private static void runTraining(DataSetBuilder ds, int[] topology,
         double lr, int epochs, int patience, long seed,
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
      IDifferentiableFunction[] act = new IDifferentiableFunction[topology.length - 1];
      int n = act.length - 1;
      for (int i = 0; i < n; ++i) {
         act[i] = new LeakyReLU();
      }
      // only last activation can be sigmoid as well since we are doing binary
      // classification
      act[n] = new Sigmoid();
      Trainer trainer = new Trainer(topology, act,
            new TrainConfig(new DataSet(trX, trY),
                  new DataSet(teX, teY), lr, epochs, epochs,
                  RandomProvider.of(seed)),
            RandomProvider.of(seed));
      System.out.println(trainer.train());
      EvaluationResult evalResult = trainer.evaluate();
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
