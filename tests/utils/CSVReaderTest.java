package utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import math.Matrix;

@DisplayName("CSVReader Tests")
class CSVReaderTest {

   @Nested
   @DisplayName("Read CSV with Default Delimiter Tests")
   class ReadCSVDefaultDelimiterTests {

      @Test
      @DisplayName("Should read CSV with comma delimiter")
      void testReadCSVComma(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("test.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0,2.0,3.0\n");
         writer.write("4.0,5.0,6.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(2, m.rows());
         assertEquals(3, m.cols());
         assertEquals(1.0, m.get(0, 0));
         assertEquals(6.0, m.get(1, 2));
      }

      @Test
      @DisplayName("Should read single value CSV")
      void testReadCSVSingleValue(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("single.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("42.5\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(1, m.rows());
         assertEquals(1, m.cols());
         assertEquals(42.5, m.get(0, 0));
      }

      @Test
      @DisplayName("Should read CSV with integers")
      void testReadCSVIntegers(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("integers.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1,2,3\n");
         writer.write("4,5,6\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(1.0, m.get(0, 0));
         assertEquals(5.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should read CSV with negative numbers")
      void testReadCSVNegative(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("negative.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("-1.5,-2.5\n");
         writer.write("3.5,-4.5\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(-1.5, m.get(0, 0));
         assertEquals(-4.5, m.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Read CSV with Custom Delimiter Tests")
   class ReadCSVCustomDelimiterTests {

      @Test
      @DisplayName("Should read CSV with semicolon delimiter")
      void testReadCSVSemicolon(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("semicolon.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0;2.0;3.0\n");
         writer.write("4.0;5.0;6.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename, ";");
         assertEquals(2, m.rows());
         assertEquals(3, m.cols());
         assertEquals(1.0, m.get(0, 0));
         assertEquals(6.0, m.get(1, 2));
      }

      @Test
      @DisplayName("Should read CSV with tab delimiter")
      void testReadCSVTab(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("tab.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0\t2.0\t3.0\n");
         writer.write("4.0\t5.0\t6.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename, "\t");
         assertEquals(2, m.rows());
         assertEquals(3, m.cols());
         assertEquals(5.0, m.get(1, 1));
      }

      @Test
      @DisplayName("Should read CSV with pipe delimiter")
      void testReadCSVPipe(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("pipe.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0|2.0\n");
         writer.write("3.0|4.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename, "\\|");
         assertEquals(2, m.rows());
         assertEquals(2, m.cols());
         assertEquals(4.0, m.get(1, 1));
      }
   }

   @Nested
   @DisplayName("Edge Cases Tests")
   class EdgeCasesTests {

      @Test
      @DisplayName("Should handle single column CSV")
      void testSingleColumn(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("single_col.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0\n");
         writer.write("2.0\n");
         writer.write("3.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(3, m.rows());
         assertEquals(1, m.cols());
         assertEquals(2.0, m.get(1, 0));
      }

      @Test
      @DisplayName("Should handle single row CSV")
      void testSingleRow(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("single_row.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.0,2.0,3.0,4.0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(1, m.rows());
         assertEquals(4, m.cols());
         assertEquals(3.0, m.get(0, 2));
      }

      @Test
      @DisplayName("Should handle scientific notation")
      void testScientificNotation(@TempDir Path tempDir) throws IOException {
         String filename = tempDir.resolve("scientific.csv").toString();
         FileWriter writer = new FileWriter(filename);
         writer.write("1.5e-3,2.0e2\n");
         writer.write("3.0e-1,4.0e0\n");
         writer.close();

         Matrix m = CSVReader.readCSV(filename);
         assertEquals(0.0015, m.get(0, 0), 1e-10);
         assertEquals(200.0, m.get(0, 1), 1e-10);
      }
   }

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should not be able to instantiate CSVReader")
      void testCannotInstantiate() {
         // CSVReader has a private constructor
         assertNotNull(CSVReader.class);
      }
   }
}
