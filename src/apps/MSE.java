package apps;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Utility class for saving Mean Squared Error (MSE) data to CSV files.
 * This class provides functionality to export training metrics for analysis and
 * visualization.
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511052002
 */
public class MSE {
   /**
    * Private constructor to prevent instantiation of this utility class.
    */
   private MSE() {
      // Prevent instantiation
   }

   /**
    * Saves an array of MSE values to a CSV file.
    * The file will contain two columns: epoch number and corresponding MSE value.
    * The format is semicolon-separated with a header row.
    *
    * @param mse      array of MSE values, where index represents the epoch number
    * @param filename the name of the file to write to (e.g., "mse.csv")
    */
   public static void saveMSE(double[] mse, String filename) {
      try (FileWriter writer = new FileWriter(filename)) {
         writer.write("epoch;mse\n");
         int n = mse.length;
         for (int i = 0; i < n; ++i) {
            writer.write(i + ";" + mse[i] + "\n");
         }
         System.out.println("MSE saved to " + filename);
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
}
