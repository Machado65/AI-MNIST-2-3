package apps;

import java.io.FileWriter;
import java.io.IOException;

public class MSE {
   private MSE() {
      // Prevent instantiation
      //
   }

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
