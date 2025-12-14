package ml.data;

import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

import math.Array;
import math.Matrix;
import utils.CSVReader;

/**
 * Builder class for loading, preprocessing, and augmenting datasets.
 * Handles data normalization, label conversion, train/test splitting, and
 * augmentation.
 * Designed for image classification tasks with MNIST-style data.
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class DataSetBuilder {
   private Matrix x;
   private Matrix y;
   private Matrix trX;
   private Matrix trY;
   private Matrix teX;
   private Matrix teY;

   /**
    * Constructs a DataSetBuilder by loading data from CSV files.
    *
    * @param datasetPath path to CSV file containing features (one sample per row)
    * @param labelsPath  path to CSV file containing labels (one label per row)
    */
   public DataSetBuilder(String datasetPath, String labelsPath) {
      this.x = CSVReader.readCSV(datasetPath);
      this.y = CSVReader.readCSV(labelsPath);
   }

   /**
    * @return the training features matrix.
    */
   public Matrix getTrX() {
      return this.trX;
   }

   /**
    * @return the training labels matrix.
    */
   public Matrix getTrY() {
      return this.trY;
   }

   /**
    * @return the testing features matrix.
    */
   public Matrix getTeX() {
      return this.teX;
   }

   /**
    * @return the testing labels matrix.
    */
   public Matrix getTeY() {
      return this.teY;
   }

   /**
    * Normalizes the dataset features to the range [0, 1].
    */
   public void normalize() {
      this.x = this.x.apply(v -> v / 255.0);
   }

   /**
    * Converts the labels using the provided function.
    *
    * @param fnc the function to apply to each label.
    */
   public void convertLabels(DoubleUnaryOperator fnc) {
      this.y = this.y.apply(fnc);
   }

   /**
    * Augments the current dataset by applying transformations to existing samples.
    * Creates multiple augmented copies of each original sample.
    * WARNING: Augments from currently stored data, which may already be augmented.
    *
    * @param copies    number of augmented copies to create per sample
    * @param transform transformation function applied to each copy
    */
   private void augment(int copies,
         BiConsumer<double[], double[]> transform) {
      int origRows = x.rows();
      int cols = x.cols();
      int totalRows = origRows * (1 + copies);
      double[][] augmentedX = new double[totalRows][cols];
      double[][] augmentedY = new double[totalRows][1];
      double[] input = new double[cols];
      double[] output = new double[cols];
      int idx = 0;
      for (int i = 0; i < origRows; ++i) {
         double label = y.get(i, 0);
         for (int j = 0; j < cols; ++j) {
            input[j] = x.get(i, j);
         }
         System.arraycopy(input, 0, augmentedX[idx], 0,
               cols);
         augmentedY[idx][0] = label;
         ++idx;
         for (int c = 0; c < copies; ++c) {
            for (int j = 0; j < cols; ++j) {
               input[j] = x.get(i, j);
            }
            transform.accept(input, output);
            System.arraycopy(output, 0, augmentedX[idx],
                  0, cols);
            augmentedY[idx][0] = label;
            ++idx;
         }
      }
      x = new Matrix(augmentedX);
      y = new Matrix(augmentedY);
   }

   /**
    * Adds Gaussian noise augmentation to the dataset.
    *
    * @param stdDev standard deviation of noise (e.g., 0.02)
    * @param copies number of noisy copies per original sample
    * @param rand   random number generator
    */
   public void addGaussianNoise(double stdDev, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.gaussianNoise(
            input, output, stdDev, rand));
   }

   /**
    * Adds brightness adjustment augmentation to the dataset.
    *
    * @param copies number of brightness-adjusted copies per sample
    * @param rand   random number generator
    */
   public void addBrightnessAdjustment(int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.applyBrightness(input, output, rand));
   }

   /**
    * Adds elastic deformation augmentation to the dataset.
    *
    * @param alpha  deformation intensity (typical: 6.0)
    * @param sigma  smoothing parameter (typical: 2.0)
    * @param copies number of deformed copies per sample
    * @param rand   random number generator
    */
   public void addElasticDeformation(double alpha, double sigma,
         int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.elasticDeform(
            input, output, rand, alpha, sigma));
   }

   /**
    * Adds rotation augmentation to the dataset.
    *
    * @param maxDegrees maximum rotation angle in degrees (e.g., 5.0)
    * @param copies     number of rotated copies per sample
    * @param rand       random number generator
    */
   public void addRotation(double maxDegrees, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.applyRotation(
            input, output, rand, maxDegrees));
   }

   /**
    * Adds pixel shift augmentation to the dataset.
    *
    * @param maxShift maximum shift in pixels (e.g., 2)
    * @param copies   number of shifted copies per sample
    * @param rand     random number generator
    */
   public void addShift(int maxShift, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.shift(
            input, output, maxShift, rand));
   }

   /**
    * Adds combined augmentation: elastic deformation + rotation + brightness.
    *
    * @param copies     number of augmented copies per sample
    * @param rand       random number generator
    * @param alpha      deformation intensity
    * @param sigma      smoothing parameter
    * @param maxDegrees maximum rotation angle
    */
   public void addCombinedAugmentation1(int copies, Random rand,
         double stdDev, double minScale, double maxScale, double minC,
         double maxC) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         double[] temp2 = new double[input.length];
         ImageAugmentation.gaussianNoise(input, temp1, stdDev, rand);
         ImageAugmentation.applyScaling(temp1, temp2, rand, minScale,
               maxScale);
         ImageAugmentation.applyContrast(temp2, output, rand, minC,
               maxC);
      });
   }

   /**
    * Adds combined augmentation: shift + rotation.
    *
    * @param copies      number of augmented copies per sample
    * @param rand        random number generator
    * @param maxDegrees  maximum rotation angle
    * @param shiftPixels maximum shift in pixels
    */
   public void addCombinedAugmentation2(int copies, Random rand,
         double maxDegrees, int shiftPixels) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         ImageAugmentation.shift(input, temp1, shiftPixels, rand);
         ImageAugmentation.applyRotation(temp1, output, rand,
               maxDegrees);
      });
   }

   /**
    * Adds combined augmentation: elastic deformation + brightness.
    *
    * @param copies number of augmented copies per sample
    * @param rand   random number generator
    * @param alpha  deformation intensity
    * @param sigma  smoothing parameter
    */
   public void addCombinedAugmentation3(int copies, Random rand,
         double alpha, double sigma) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         ImageAugmentation.elasticDeform(input, temp1, rand, alpha,
               sigma);
         ImageAugmentation.applyBrightness(temp1, output, rand);
      });
   }

   /**
    * Splits the dataset into training and testing sets.
    * Randomly shuffles the data before splitting.
    *
    * @param trainRatio proportion of data for training (e.g., 0.8 for 80%)
    * @param rand       random number generator for shuffling
    */
   public void split(double trainRatio, Random rand) {
      int n = this.x.rows();
      int m = this.x.cols();
      int trainSize = (int) (n * trainRatio);
      int testSize = n - trainSize;
      Array indices = new Array(n);
      indices.initSequential(n);
      indices.shuffleArray(rand);
      double[][] trXData = new double[trainSize][m];
      double[][] trYData = new double[trainSize][1];
      double[][] teXData = new double[testSize][m];
      double[][] teYData = new double[testSize][1];
      for (int i = 0; i < trainSize; ++i) {
         int idx = indices.get(i);
         for (int j = 0; j < m; ++j) {
            trXData[i][j] = this.x.get(idx, j);
         }
         trYData[i][0] = this.y.get(idx, 0);
      }
      for (int i = 0; i < testSize; ++i) {
         int idx = indices.get(trainSize + i);
         for (int j = 0; j < m; ++j) {
            teXData[i][j] = this.x.get(idx, j);
         }
         teYData[i][0] = this.y.get(idx, 0);
      }
      if (trainSize > 0) {
         this.trX = new Matrix(trXData);
         this.trY = new Matrix(trYData);
      } else {
         this.trX = null;
         this.trY = null;
      }
      if (testSize > 0) {
         this.teX = new Matrix(teXData);
         this.teY = new Matrix(teYData);
      } else {
         this.teX = null;
         this.teY = null;
      }
   }
}
