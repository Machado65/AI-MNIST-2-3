package neural.activation;

import java.util.function.DoubleUnaryOperator;

public class ReLU implements IDifferentiableFunction {
   @Override
   public DoubleUnaryOperator fnc() {
      return x -> Math.max(0.0, x);
   }

   @Override
   public DoubleUnaryOperator derivative() {
      return x -> (x > 0.0) ? 1.0 : 0.01;
   }
}
