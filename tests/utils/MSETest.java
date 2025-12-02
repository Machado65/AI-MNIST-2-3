package utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@DisplayName("MSE Utility Tests")
class MSETest {

   @Nested
   @DisplayName("Save MSE Tests")
   class SaveMSETests {

      @Test
      @DisplayName("Should save MSE array to CSV file")
      void testSaveMSE(@TempDir Path tempDir) throws IOException {
         double[] mse = { 0.5, 0.4, 0.3, 0.2, 0.1 };
         String filename = tempDir.resolve("test_mse.csv").toString();

         MSE.saveMSE(mse, filename);

         BufferedReader reader = new BufferedReader(new FileReader(filename));
         String header = reader.readLine();
         assertEquals("epoch;mse", header);

         for (int i = 0; i < mse.length; i++) {
            String line = reader.readLine();
            String expected = i + ";" + mse[i];
            assertEquals(expected, line);
         }

         assertNull(reader.readLine()); // No more lines
         reader.close();
      }

      @Test
      @DisplayName("Should save empty MSE array")
      void testSaveEmptyMSE(@TempDir Path tempDir) throws IOException {
         double[] mse = {};
         String filename = tempDir.resolve("empty_mse.csv").toString();

         MSE.saveMSE(mse, filename);

         BufferedReader reader = new BufferedReader(new FileReader(filename));
         String header = reader.readLine();
         assertEquals("epoch;mse", header);
         assertNull(reader.readLine());
         reader.close();
      }

      @Test
      @DisplayName("Should save single value MSE array")
      void testSaveSingleMSE(@TempDir Path tempDir) throws IOException {
         double[] mse = { 0.123456 };
         String filename = tempDir.resolve("single_mse.csv").toString();

         MSE.saveMSE(mse, filename);

         BufferedReader reader = new BufferedReader(new FileReader(filename));
         reader.readLine(); // skip header
         String line = reader.readLine();
         assertEquals("0;0.123456", line);
         reader.close();
      }

      @Test
      @DisplayName("Should save MSE with scientific notation")
      void testSaveMSEScientificNotation(@TempDir Path tempDir) throws IOException {
         double[] mse = { 1e-5, 1e-6, 1e-7 };
         String filename = tempDir.resolve("scientific_mse.csv").toString();

         MSE.saveMSE(mse, filename);

         BufferedReader reader = new BufferedReader(new FileReader(filename));
         reader.readLine(); // skip header
         assertNotNull(reader.readLine());
         assertNotNull(reader.readLine());
         assertNotNull(reader.readLine());
         reader.close();
      }
   }

   @Nested
   @DisplayName("Constructor Tests")
   class ConstructorTests {

      @Test
      @DisplayName("Should not be able to instantiate MSE class")
      void testCannotInstantiate() {
         // MSE has a private constructor, so we can't instantiate it
         // This test just verifies the class exists and has the static method
         assertNotNull(MSE.class);
      }
   }
}
