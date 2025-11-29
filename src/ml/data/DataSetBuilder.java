package ml.data;

import java.util.Random;
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
    * Converts the labels using the provided function.
    *
    * @param fnc the function to apply to each label.
    */
   public void convertLabels(DoubleUnaryOperator fnc) {
      this.y = this.y.apply(fnc);
   }

   /**
    * Adds Gaussian noise to create augmented copies of the images.
    *
    * @param stdDev standard deviation of Gaussian noise (e.g., 0.1)
    * @param copies number of noisy copies per original image
    */
   public void addGaussianNoise(double stdDev, int copies, Random rand) {
      int origRows = this.x.rows();
      int cols = this.x.cols();
      int totalRows = origRows * (1 + copies);
      double[][] augmentedX = new double[totalRows][cols];
      double[][] augmentedY = new double[totalRows][1];
      int idx = 0;
      for (int i = 0; i < origRows; ++i) {
         double origLabel = this.y.get(i, 0);
         for (int j = 0; j < cols; ++j) {
            augmentedX[idx][j] = this.x.get(i, j);
         }
         augmentedY[idx][0] = origLabel;
         ++idx;
         for (int c = 0; c < copies; ++c) {
            for (int j = 0; j < cols; ++j) {
               double noisy = this.x.get(i, j) +
                     rand.nextGaussian() * stdDev;
               augmentedX[idx][j] = Math.clamp(noisy, 0.0, 1.0);
            }
            augmentedY[idx][0] = origLabel;
            ++idx;
         }
      }
      this.x = new Matrix(augmentedX);
      this.y = new Matrix(augmentedY);
   }

   public void split(double trainRatio, Random rand) {
      int n = this.x.rows();
      int m = this.x.cols();
      int trainSize = (int) (n * trainRatio);
      int testSize = n - trainSize;
      Array indices = new Array(n);
      indices.initializeSequential(n);
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
      this.trX = new Matrix(trXData);
      this.trY = new Matrix(trYData);
      this.teX = new Matrix(teXData);
      this.teY = new Matrix(teYData);
   }
}
