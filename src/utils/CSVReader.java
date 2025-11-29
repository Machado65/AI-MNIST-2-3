package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import math.Matrix;

/**
 * Utility class for reading CSV files and converting them to Matrix objects.
 *
 * @author André Martins, António Matoso, Tomás Machado
 */
public class CSVReader {
   private static final String DEFAULT_DELIMITER = ",";

   private CSVReader() {
      // Prevent instantiation
   }

   /**
    * Reads a CSV file and converts it to a Matrix.
    * Each line becomes a row in the matrix.
    *
    * @param filename  path to the CSV file
    * @param delimiter delimiter used (e.g., "," or ";")
    * @return Matrix with the CSV data
    */
   public static Matrix readCSV(String filename, String delimiter) {
      List<double[]> rows = new ArrayList<>();
      try (BufferedReader br = new BufferedReader(
            new FileReader(filename))) {
         String line;
         while ((line = br.readLine()) != null) {
            String[] values = line.split(delimiter);
            int n = values.length;
            double[] row = new double[n];
            for (int i = 0; i < n; ++i) {
               row[i] = Double.parseDouble(values[i]);
            }
            rows.add(row);
         }
      } catch (IOException | NumberFormatException e) {
         e.printStackTrace();
      }
      return new Matrix(rows);
   }

   /**
    * Reads a CSV file with comma delimiter.
    *
    * @param filename path to the CSV file
    * @return Matrix with the CSV data
    */
   public static Matrix readCSV(String filename) {
      return readCSV(filename, DEFAULT_DELIMITER);
   }
}
