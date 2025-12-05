package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import utils.RandomProvider;

public class MLPConfig {
   private static final int INPUT_SIZE = 400;
   private static final String DATASET_PATH = "src/ml/data/dataset.csv";
   private static final String LABELS_PATH = "src/ml/data/labels.csv";
   private static final long[] seeds = { 1, 12, 123, 2024, 999 };

   public static void main(String[] args) {
      for (long seed : seeds) {
         System.out.println("\n\n=== RUNNING WITH SEED: " +
               seed + " ===");
         config1Baseline(seed);
         config2Normalized(seed);
         config3MoreNeurons(seed);
         config4Deeper(seed);
         config5LowerLr(seed);
         config6WithNoise(seed);
         config7Combo1(seed);
         config8Combo2(seed);
         config9Aggressive(seed);
         config10Ultra(seed);
      }
   }

   private static void config1Baseline(long seed) {
      System.out.println("\n=== CONFIG 1: BASELINE ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 32, 1 }, 0.05, 10000,
            200, seed, "config1");
   }

   private static void config2Normalized(long seed) {
      System.out.println("\n=== CONFIG 2: WITH NORMALIZATION ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 32, 1 }, 0.03, 10000,
            200, seed, "config2");
   }

   private static void config3MoreNeurons(long seed) {
      System.out.println("\n=== CONFIG 3: MORE NEURONS (64) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 64, 1 }, 0.03, 10000,
            200, seed, "config3");
   }

   private static void config4Deeper(long seed) {
      System.out.println("\n=== CONFIG 4: DEEPER NETWORK (64-32) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 64, 32, 1 }, 0.05,
            15000, 300, seed, "config4");
   }

   private static void config5LowerLr(long seed) {
      System.out.println("\n=== CONFIG 5: LOWER LR (0.05) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 32, 1 }, 0.05, 15000,
            300, seed, "config5");
   }

   private static void config6WithNoise(long seed) {
      System.out.println("\n=== CONFIG 6: WITH GAUSSIAN NOISE ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.addGaussianNoise(0.01, 1, RandomProvider.of(seed)); // ATIVADO
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 32, 1 }, 0.05, 15000,
            300, seed, "config6");
   }

   private static void config7Combo1(long seed) {
      System.out.println("\n=== CONFIG 7: COMBO 1 (Norm + 48 neurons + LR 0.05) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 48, 1 }, 0.05, 15000,
            300, seed, "config7");
   }

   private static void config8Combo2(long seed) {
      System.out.println("\n=== CONFIG 8: COMBO 2 (Norm + Noise + 64 neurons) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.addGaussianNoise(0.005, 1,
            RandomProvider.of(seed));
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 64, 1 }, 0.05, 15000,
            300, seed, "config8");
   }

   private static void config9Aggressive(long seed) {
      System.out.println("\n=== CONFIG 9: AGGRESSIVE (128 neurons, LR 0.03) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.03,
            20000, 400, seed, "config9");
   }

   private static void config10Ultra(long seed) {
      System.out.println("\n=== CONFIG 10: ULTRA (80-40-20, LR 0.02) ===");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.normalize();
      ds.addGaussianNoise(0.005, 1, RandomProvider.of(seed));
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 80, 40, 20, 1 },
            0.02, 25000, 500, seed,
            "config10");
   }

   private static void runTraining(DataSetBuilder ds, int[] topology,
         double lr, int epochs, int patience, long seed, String configName) {
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      IDifferentiableFunction[] activations = new IDifferentiableFunction[topology.length - 1];
      for (int i = 0; i < activations.length; ++i) {
         activations[i] = new Sigmoid();
      }
      Trainer trainer = new Trainer(
            topology,
            lr,
            epochs,
            patience,
            activations,
            RandomProvider.of(seed));
      System.out.println(trainer.train(trX, trY, teX, teY));
      System.out.println(trainer.evaluate(teX, teY));
      try {
         trainer.getMLP().saveModel("src/ml/models/mlp_" +
               configName + ".dat");
         System.out.println("Model saved: mlp_" + configName + ".dat");
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
