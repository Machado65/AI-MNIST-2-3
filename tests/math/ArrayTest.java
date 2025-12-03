package math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Array Tests")
class ArrayTest {

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should create array with specified size")
      void testConstructor() {
         Array arr = new Array(5);
         assertNotNull(arr);
      }
   }

   @Nested
   @DisplayName("Initialization Tests")
   class InitializationTests {

      @Test
      @DisplayName("Should initialize array sequentially")
      void testInitializeSequential() {
         Array arr = new Array(5);
         arr.initSequential(5);
         assertEquals(0, arr.get(0));
         assertEquals(1, arr.get(1));
         assertEquals(4, arr.get(4));
      }

      @Test
      @DisplayName("Should initialize with correct values")
      void testInitializeSequentialValues() {
         Array arr = new Array(10);
         arr.initSequential(10);
         for (int i = 0; i < 10; i++) {
            assertEquals(i, arr.get(i));
         }
      }
   }

   @Nested
   @DisplayName("Get and Swap Tests")
   class GetSwapTests {

      @Test
      @DisplayName("Should get element at index")
      void testGet() {
         Array arr = new Array(3);
         arr.initSequential(3);
         assertEquals(0, arr.get(0));
         assertEquals(2, arr.get(2));
      }

      @Test
      @DisplayName("Should swap two elements")
      void testSwap() {
         Array arr = new Array(3);
         arr.initSequential(3);
         arr.swap(0, 2);
         assertEquals(2, arr.get(0));
         assertEquals(0, arr.get(2));
         assertEquals(1, arr.get(1));
      }

      @Test
      @DisplayName("Should swap same element (no change)")
      void testSwapSameIndex() {
         Array arr = new Array(3);
         arr.initSequential(3);
         arr.swap(1, 1);
         assertEquals(1, arr.get(1));
      }
   }

   @Nested
   @DisplayName("Shuffle Tests")
   class ShuffleTests {

      @Test
      @DisplayName("Should shuffle array")
      void testShuffleArray() {
         Array arr = new Array(10);
         arr.initSequential(10);
         Random rand = new Random(42);
         arr.shuffleArray(rand);

         // Verify all elements are still present
         boolean[] found = new boolean[10];
         for (int i = 0; i < 10; i++) {
            int val = arr.get(i);
            assertTrue(val >= 0 && val < 10);
            found[val] = true;
         }
         for (boolean b : found) {
            assertTrue(b);
         }
      }

      @Test
      @DisplayName("Should produce different shuffle with different seed")
      void testShuffleDifferentSeeds() {
         Array arr1 = new Array(10);
         arr1.initSequential(10);
         Array arr2 = new Array(10);
         arr2.initSequential(10);

         Random rand1 = new Random(42);
         Random rand2 = new Random(123);

         arr1.shuffleArray(rand1);
         arr2.shuffleArray(rand2);

         // Arrays should be different (with very high probability)
         boolean different = false;
         for (int i = 0; i < 10; i++) {
            if (arr1.get(i) != arr2.get(i)) {
               different = true;
               break;
            }
         }
         assertTrue(different);
      }

      @Test
      @DisplayName("Should shuffle consistently with same seed")
      void testShuffleSameSeed() {
         Array arr1 = new Array(10);
         arr1.initSequential(10);
         Array arr2 = new Array(10);
         arr2.initSequential(10);

         Random rand1 = new Random(42);
         Random rand2 = new Random(42);

         arr1.shuffleArray(rand1);
         arr2.shuffleArray(rand2);

         // Arrays should be identical
         for (int i = 0; i < 10; i++) {
            assertEquals(arr1.get(i), arr2.get(i));
         }
      }
   }
}
