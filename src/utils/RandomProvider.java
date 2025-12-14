package utils;

import java.util.Random;

/**
 * Utility class providing various Random instance factories.
 * Offers fixed-seed, custom-seed, and global random number generators.
 * Used throughout the application for reproducible experiments.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class RandomProvider {
   private static final long SEED = 42;
   private static final Random FIXED = new Random(SEED);
   private static final Random GLOBAL = new Random();

   private RandomProvider() {
      // Prevent instantiation
   }

   /**
    * Creates a new Random instance with the specified seed.
    * Useful for reproducible experiments with custom seeds.
    *
    * @param seed the seed value for the random number generator
    * @return a new Random instance initialized with the given seed
    */
   public static Random of(long seed) {
      return new Random(seed);
   }

   /**
    * Returns a fixed Random instance with seed 42.
    * Use this for completely reproducible results across runs.
    *
    * @return a fixed Random instance (seed=42)
    */
   public static Random fixed() {
      return FIXED;
   }

   /**
    * Returns a global Random instance with unpredictable seed.
    * Use this when randomness is desired without reproducibility.
    *
    * @return a global Random instance
    */
   public static Random global() {
      return GLOBAL;
   }
}
