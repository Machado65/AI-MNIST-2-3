package utils;

import java.util.Random;

public class RandomProvider {
   private static final long SEED = 42;
   private static final Random FIXED = new Random(SEED);
   private static final Random GLOBAL = new Random();

   private RandomProvider() {
      // Prevent instantiation
   }

   public static Random fixed() {
      return FIXED;
   }

   public static Random global() {
      return GLOBAL;
   }
}
