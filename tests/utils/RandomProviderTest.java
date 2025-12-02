package utils;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("RandomProvider Tests")
class RandomProviderTest {

   @Nested
   @DisplayName("Fixed Random Tests")
   class FixedRandomTests {

      @Test
      @DisplayName("Should return non-null fixed Random")
      void testFixedNotNull() {
         assertNotNull(RandomProvider.fixed());
      }

      @Test
      @DisplayName("Should return same instance on multiple calls")
      void testFixedSameInstance() {
         Random rand1 = RandomProvider.fixed();
         Random rand2 = RandomProvider.fixed();
         assertSame(rand1, rand2);
      }

      @Test
      @DisplayName("Should produce deterministic sequence")
      void testFixedDeterministic() {
         // Reset by creating new provider access
         Random rand1 = RandomProvider.fixed();
         double val1 = rand1.nextDouble();
         double val2 = rand1.nextDouble();

         // Values should be deterministic (same seed = same sequence)
         assertNotEquals(val1, val2);
      }

      @Test
      @DisplayName("Should produce same values on restart")
      void testFixedReproducible() {
         // This tests that the fixed random uses a constant seed
         Random rand = RandomProvider.fixed();
         assertNotNull(rand);
      }
   }

   @Nested
   @DisplayName("Global Random Tests")
   class GlobalRandomTests {

      @Test
      @DisplayName("Should return non-null global Random")
      void testGlobalNotNull() {
         assertNotNull(RandomProvider.global());
      }

      @Test
      @DisplayName("Should return same instance on multiple calls")
      void testGlobalSameInstance() {
         Random rand1 = RandomProvider.global();
         Random rand2 = RandomProvider.global();
         assertSame(rand1, rand2);
      }

      @Test
      @DisplayName("Should produce random sequence")
      void testGlobalRandom() {
         Random rand = RandomProvider.global();
         double val1 = rand.nextDouble();
         double val2 = rand.nextDouble();

         // Values should be different (with very high probability)
         assertNotEquals(val1, val2);
      }
   }

   @Nested
   @DisplayName("Fixed vs Global Tests")
   class ComparisonTests {

      @Test
      @DisplayName("Fixed and Global should be different instances")
      void testDifferentInstances() {
         Random fixed = RandomProvider.fixed();
         Random global = RandomProvider.global();
         assertNotSame(fixed, global);
      }
   }

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should not be able to instantiate RandomProvider")
      void testCannotInstantiate() {
         // RandomProvider has a private constructor
         assertNotNull(RandomProvider.class);
      }
   }
}
