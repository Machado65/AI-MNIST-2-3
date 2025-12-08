package apps;

import math.Matrix;
import ml.data.DataSetBuilder;
import ml.training.EvaluationResult;
import ml.training.Trainer;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import utils.RandomProvider;

/**
 * Optimized configurations based on best results:
 * Baseline: 400-128-1, LeakyReLU+Sigmoid, LR=0.005, Epochs=15000, Patience=800
 *
 * Improvements to try:
 * 1. Data Augmentation (Elastic deformation works best for MNIST)
 * 2. Deeper networks with residual-like connections
 * 3. Different topologies (wider vs deeper)
 * 4. Learning rate scheduling (decay)
 * 5. Batch normalization equivalent (layer normalization)
 * 6. Dropout simulation (noise injection)
 * 7. Different activation combinations
 */
public class OptimizedConfigs {
   private static final String DATASET_PATH = "data/largeDataset.csv";
   private static final String LABELS_PATH = "data/largeLabels.csv";
   private static final long[] SEEDS = { 123, 456, 789, 2024 };

   public static void main(String[] args) {
      for (long seed : SEEDS) {
         System.out.println("\n" + "=".repeat(60));
         System.out.println("SEED: " + seed);
         System.out.println("=".repeat(60));
         // Baseline (your best)
         runBaseline(seed);
         // Strategy 1: Data Augmentation
         runWithElasticDeformation(seed);
         runWithCombinedAugmentation(seed);
         // Strategy 2: Architecture improvements
         runDeeperNetwork(seed);
         runWiderNetwork(seed);
         runPyramidTopology(seed);
         // Strategy 3: Training tricks
         runWithLowerLR(seed);
         runWithHigherPatience(seed);
      }
   }

   // ====================================================================
   // BASELINE (Your current best)
   // ====================================================================

   private static void runBaseline(long seed) {
      System.out.println("\n>>> BASELINE (400-128-1)");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.005, 15000, 800,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "baseline");
   }

   // ====================================================================
   // STRATEGY 1: DATA AUGMENTATION
   // ====================================================================

   /**
    * Elastic deformation - proven to work extremely well for MNIST.
    * Paper: Simard et al. "Best Practices for CNNs"
    * Expected improvement: +1-3% accuracy
    */
   private static void runWithElasticDeformation(long seed) {
      System.out.println("\n>>> WITH ELASTIC DEFORMATION");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      // Elastic: alpha=8, sigma=3 (paper recommendations)
      ds.addElasticDeformation(8.0, 3.0, 1,
            RandomProvider.of(seed));
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.004, 18000, 1000,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "elastic");
   }

   /**
    * Combined augmentation: elastic + rotation + brightness.
    * Most robust, prevents overfitting.
    * Expected improvement: +2-4% accuracy, better generalization
    */
   private static void runWithCombinedAugmentation(long seed) {
      System.out.println("\n>>> WITH COMBINED AUGMENTATION");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.addCombinedAugmentation1(1, RandomProvider.of(seed),
            8.0, 3.0, 10.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.003, 20000, 1200,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "combined_aug");
   }

   // ====================================================================
   // STRATEGY 2: ARCHITECTURE IMPROVEMENTS
   // ====================================================================

   /**
    * Deeper network: 400 → 128 → 64 → 32 → 1
    * More layers = more capacity to learn complex patterns.
    * Expected: Better feature learning, may need more epochs
    */
   private static void runDeeperNetwork(long seed) {
      System.out.println("\n>>> DEEPER NETWORK (4 hidden layers)");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 64, 32, 1 },
            0.003, 20000, 1000,
            new IDifferentiableFunction[] {
                  new LeakyReLU(), new LeakyReLU(),
                  new LeakyReLU(), new Sigmoid()
            },
            seed, "deeper");
   }

   /**
    * Wider network: 400 → 256 → 1
    * More neurons per layer = more representational power.
    * Expected: Faster convergence, may overfit easier
    */
   private static void runWiderNetwork(long seed) {
      System.out.println("\n>>> WIDER NETWORK (256 neurons)");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 256, 1 }, 0.003, 15000, 800,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "wider");
   }

   /**
    * Pyramid topology: 400 → 192 → 96 → 48 → 1
    * Gradual dimension reduction, like a funnel.
    * Expected: Smooth feature compression, good balance
    */
   private static void runPyramidTopology(long seed) {
      System.out.println("\n>>> PYRAMID TOPOLOGY");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 192, 96, 48, 1 },
            0.003, 20000, 1000,
            new IDifferentiableFunction[] {
                  new LeakyReLU(), new LeakyReLU(),
                  new LeakyReLU(), new Sigmoid()
            },
            seed, "pyramid");
   }

   // ====================================================================
   // STRATEGY 3: TRAINING TRICKS
   // ====================================================================

   /**
    * Lower learning rate for more stable convergence.
    * Expected: More stable training, slightly better final accuracy
    */
   private static void runWithLowerLR(long seed) {
      System.out.println("\n>>> LOWER LEARNING RATE (0.003)");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.003, 20000, 1000,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "lower_lr");
   }

   /**
    * Higher patience to avoid premature stopping.
    * Expected: Find better local minima, more consistent results
    */
   private static void runWithHigherPatience(long seed) {
      System.out.println("\n>>> HIGHER PATIENCE (1500)");
      DataSetBuilder ds = new DataSetBuilder(DATASET_PATH, LABELS_PATH);
      ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
      ds.split(0.8, RandomProvider.of(seed));
      runTraining(ds, new int[] { 400, 128, 1 }, 0.005, 20000, 1500,
            new IDifferentiableFunction[] { new LeakyReLU(), new Sigmoid() },
            seed, "high_patience");
   }

   // ====================================================================
   // HELPER METHOD
   // ====================================================================

   private static void runTraining(DataSetBuilder ds, int[] topology,
         double lr, int epochs, int patience,
         IDifferentiableFunction[] activations,
         long seed, String configName) {
      Matrix trX = ds.getTrX();
      Matrix trY = ds.getTrY();
      Matrix teX = ds.getTeX();
      Matrix teY = ds.getTeY();
      Trainer trainer = new Trainer(topology, lr, epochs, patience,
            activations, RandomProvider.of(seed));
      System.out.println(trainer.train(trX, trY, teX, teY));
      EvaluationResult evalResult = trainer.evaluate(teX, teY);
      System.out.println(evalResult);
      // Only save if accuracy is good
      if (evalResult.getAccuracy() >= 0.97) {
         try {
            String modelName = String.format("mlp_%s_seed%d", configName, seed);
            trainer.getMLP().saveModel("src/ml/models/" + modelName + ".dat",
                  evalResult.getOptimalThreshold());
            System.out.println("✓ Model saved: " + modelName);
         } catch (Exception e) {
            System.err.println("✗ Failed to save model: " + e.getMessage());
         }
      } else {
         System.out.println("✗ Accuracy too low, model not saved");
      }
   }
}
