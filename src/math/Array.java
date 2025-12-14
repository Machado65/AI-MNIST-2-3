package math;

import java.util.Random;

/**
 * Array utility class for managing integer arrays.
 * Provides functionality for initialization, shuffling, and element access.
 * Used primarily for managing indices in batch training.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class Array {
   private int[] data;

   /**
    * Constructs an array of specified size.
    *
    * @param n the size of the array
    */
   public Array(int n) {
      this.data = new int[n];
   }

   /**
    * Initializes the array with sequential values from 0 to n-1.
    * Example: for n=5, array becomes [0, 1, 2, 3, 4]
    *
    * @param n the number of sequential values to initialize
    */
   public void initSequential(int n) {
      for (int i = 0; i < n; ++i) {
         this.data[i] = i;
      }
   }

   /**
    * Gets the element at index i.
    *
    * @param i index to get.
    * @return the element at index i.
    */
   public int get(int i) {
      return this.data[i];
   }

   /**
    * Swaps the elements at indices i and j in the array.
    *
    * @param i the index of the first element to swap.
    * @param j the index of the second element to swap.
    */
   public void swap(int i, int j) {
      int temp = this.data[i];
      this.data[i] = this.data[j];
      this.data[j] = temp;
   }

   /**
    * Shuffles the array in place using the Fisher-Yates algorithm.
    *
    * @param seed the seed for the random number generator,
    *             if negative, a variable seed is used.
    */
   public void shuffleArray(Random rand) {
      for (int i = this.data.length - 1; i > 0; --i) {
         this.swap(i, rand.nextInt(i + 1));
      }
   }
}
