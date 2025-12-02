package ml.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import math.Matrix;

@DisplayName("DataSetBuilder Tests")
class DataSetBuilderTest {

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should load dataset and labels from CSV files")
      void testConstructor(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2,3\n4,5,6\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n1\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         assertNotNull(ds);
      }
   }

   @Nested
   @DisplayName("Normalize Tests")
   class NormalizeTests {

      @Test
      @DisplayName("Should normalize values to [0, 1] range")
      void testNormalize(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("0,127.5,255\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.normalize();
         ds.split(1.0, new Random(42));

         Matrix trX = ds.getTrX();
         assertEquals(0.0, trX.get(0, 0), 0.0001);
         assertEquals(0.5, trX.get(0, 1), 0.0001);
         assertEquals(1.0, trX.get(0, 2), 0.0001);
      }

      @Test
      @DisplayName("Should handle already normalized data")
      void testNormalizeAlreadyNormalized(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("0.5,0.8\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.normalize();
         ds.split(1.0, new Random(42));

         Matrix trX = ds.getTrX();
         assertTrue(trX.get(0, 0) < 0.01);
         assertTrue(trX.get(0, 1) < 0.01);
      }
   }

   @Nested
   @DisplayName("Convert Labels Tests")
   class ConvertLabelsTests {

      @Test
      @DisplayName("Should convert labels using provided function")
      void testConvertLabels(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2\n3,4\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("2\n3\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.convertLabels(label -> (label == 2.0) ? 0.0 : 1.0);
         ds.split(1.0, new Random(42));

         Matrix trY = ds.getTrY();
         assertEquals(0.0, trY.get(0, 0));
         assertEquals(1.0, trY.get(1, 0));
      }

      @Test
      @DisplayName("Should convert binary labels")
      void testConvertBinaryLabels(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("5\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.convertLabels(label -> label > 3 ? 1.0 : 0.0);
         ds.split(1.0, new Random(42));

         assertEquals(1.0, ds.getTrY().get(0, 0));
      }
   }

   @Nested
   @DisplayName("Add Gaussian Noise Tests")
   class GaussianNoiseTests {

      @Test
      @DisplayName("Should augment dataset with noisy copies")
      void testAddGaussianNoise(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("0.5,0.5\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("1\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.addGaussianNoise(0.1, 2, new Random(42));
         ds.split(1.0, new Random(42));

         // Should have 1 original + 2 copies = 3 total
         assertEquals(3, ds.getTrX().rows());
         assertEquals(3, ds.getTrY().rows());
      }

      @Test
      @DisplayName("Should clamp noisy values to [0, 1]")
      void testGaussianNoiseClamp(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1.0,0.0\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("1\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.addGaussianNoise(0.5, 10, new Random(42));
         ds.split(1.0, new Random(42));

         Matrix trX = ds.getTrX();
         for (int i = 0; i < trX.rows(); i++) {
            for (int j = 0; j < trX.cols(); j++) {
               assertTrue(trX.get(i, j) >= 0.0 && trX.get(i, j) <= 1.0);
            }
         }
      }

      @Test
      @DisplayName("Should preserve labels in augmented data")
      void testGaussianNoisePreservesLabels(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("0.5,0.5\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("7\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.addGaussianNoise(0.05, 3, new Random(42));
         ds.split(1.0, new Random(42));

         Matrix trY = ds.getTrY();
         for (int i = 0; i < trY.rows(); i++) {
            assertEquals(7.0, trY.get(i, 0));
         }
      }
   }

   @Nested
   @DisplayName("Split Tests")
   class SplitTests {

      @Test
      @DisplayName("Should split dataset into train and test sets")
      void testSplit(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         for (int i = 0; i < 100; i++) {
            dataWriter.write("1,2,3\n");
         }
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         for (int i = 0; i < 100; i++) {
            labelWriter.write("1\n");
         }
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.split(0.8, new Random(42));

         assertEquals(80, ds.getTrX().rows());
         assertEquals(20, ds.getTeX().rows());
         assertEquals(80, ds.getTrY().rows());
         assertEquals(20, ds.getTeY().rows());
      }

      @Test
      @DisplayName("Should handle 100% train split")
      void testSplitAllTrain(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2\n3,4\n5,6\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n1\n2\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.split(1.0, new Random(42));

         assertEquals(3, ds.getTrX().rows());
         assertNull(ds.getTeX());
      }

      @Test
      @DisplayName("Should shuffle data when splitting")
      void testSplitShuffles(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         for (int i = 0; i < 10; i++) {
            dataWriter.write(i + "," + i + "\n");
         }
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         for (int i = 0; i < 10; i++) {
            labelWriter.write(i + "\n");
         }
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.split(0.7, new Random(42));

         // Verify data was shuffled (not in original order)
         Matrix trX = ds.getTrX();
         boolean isShuffled = false;
         for (int i = 1; i < trX.rows(); i++) {
            if (trX.get(i, 0) < trX.get(i - 1, 0)) {
               isShuffled = true;
               break;
            }
         }
         assertTrue(isShuffled || trX.rows() <= 1);
      }
   }

   @Nested
   @DisplayName("Getter Tests")
   class GetterTests {

      @Test
      @DisplayName("Should return null before split")
      void testGettersBeforeSplit(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);

         assertNull(ds.getTrX());
         assertNull(ds.getTrY());
         assertNull(ds.getTeX());
         assertNull(ds.getTeY());
      }

      @Test
      @DisplayName("Should return matrices after split")
      void testGettersAfterSplit(@TempDir Path tempDir) throws IOException {
         String datasetPath = tempDir.resolve("dataset.csv").toString();
         String labelsPath = tempDir.resolve("labels.csv").toString();

         FileWriter dataWriter = new FileWriter(datasetPath);
         dataWriter.write("1,2\n3,4\n");
         dataWriter.close();

         FileWriter labelWriter = new FileWriter(labelsPath);
         labelWriter.write("0\n1\n");
         labelWriter.close();

         DataSetBuilder ds = new DataSetBuilder(datasetPath, labelsPath);
         ds.split(0.5, new Random(42));

         assertNotNull(ds.getTrX());
         assertNotNull(ds.getTrY());
         assertNotNull(ds.getTeX());
         assertNotNull(ds.getTeY());
      }
   }
}
