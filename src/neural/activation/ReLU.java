package neural.activation;

import java.util.function.DoubleUnaryOperator;

public class ReLU implements IDifferentiableFunction {
   @Override
   public DoubleUnaryOperator fnc() {
      return x -> (x > 0) ? x : 0;
   }

   @Override
   public DoubleUnaryOperator derivative() {
      return x -> (x > 0.0) ? 1.0 : 0.0;
   }
}
