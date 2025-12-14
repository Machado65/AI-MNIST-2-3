import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import math.Matrix;
import neural.MLP;

public class P4 {
   private static final int INPUT_SIZE = 400;

   public static void main(String[] args) {
      try (BufferedReader br = new BufferedReader(
            new InputStreamReader(System.in))) {
         MLP mlp = new MLP("src/ml/models/mlp_config1s2023_C1_C2_C3_medium.dat");
         String line;
         List<Matrix> pred = new ArrayList<>();
         while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            double[][] input = new double[1][INPUT_SIZE];
            for (int i = 0; i < INPUT_SIZE; ++i) {
               input[0][i] = Double.parseDouble(values[i]);
            }
            pred.add(mlp.predict(new Matrix(input))
                  .apply(v -> (v < mlp.getOptimalThreshold()
                        .getThreshold()) ? 2 : 3));
         }
         for (Matrix p : pred) {
            System.out.print(p.toIntString());
         }
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
}
