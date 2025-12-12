package neural.activation;

import java.util.function.DoubleUnaryOperator;

public class LeakyReLU implements IDifferentiableFunction {
   private static final double ALPHA = 0.01;

   @Override
   public DoubleUnaryOperator fnc() {
      return x -> (x > 0) ? x : ALPHA * x;
   }

   @Override

   public DoubleUnaryOperator derivative() {
      return x -> (x > 0) ? 1.0 : ALPHA;
   }
}
