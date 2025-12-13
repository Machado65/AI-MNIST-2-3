package ml.data;

import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

import math.Array;
import math.Matrix;
import utils.CSVReader;

public class DataSetBuilder {
   private Matrix x;
   private Matrix y;
   private Matrix trX;
   private Matrix trY;
   private Matrix teX;
   private Matrix teY;

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

   public void addGaussianNoise(double stdDev, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.gaussianNoise(
            input, output, stdDev, rand));
   }

   public void addBrightnessAdjustment(int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.applyBrightness(input, output, rand));
   }

   public void addElasticDeformation(double alpha, double sigma,
         int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.elasticDeform(
            input, output, rand, alpha, sigma));
   }

   public void addRotation(double maxDegrees, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.applyRotation(
            input, output, rand, maxDegrees));
   }

   public void addShift(int maxShift, int copies, Random rand) {
      augment(copies, (input, output) -> ImageAugmentation.shift(
            input, output, maxShift, rand));
   }

   public void addCombinedAugmentation1(int copies, Random rand,
         double alpha, double sigma, double maxDegrees) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         double[] temp2 = new double[input.length];
         ImageAugmentation.elasticDeform(input, temp1, rand, alpha,
               sigma);
         ImageAugmentation.applyRotation(temp1, temp2, rand,
               maxDegrees);
         ImageAugmentation.applyBrightness(temp2, output, rand);
      });
   }

   public void addCombinedAugmentation2(int copies, Random rand,
         double maxDegrees, int shiftPixels) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         ImageAugmentation.shift(input, temp1, shiftPixels, rand);
         ImageAugmentation.applyRotation(temp1, output, rand,
               maxDegrees);
      });
   }

   public void addCombinedAugmentation3(int copies, Random rand,
         double alpha, double sigma) {
      augment(copies, (input, output) -> {
         double[] temp1 = new double[input.length];
         ImageAugmentation.elasticDeform(input, temp1, rand, alpha,
               sigma);
         ImageAugmentation.applyBrightness(temp1, output, rand);
      });
   }

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
