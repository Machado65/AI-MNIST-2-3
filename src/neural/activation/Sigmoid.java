package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511101225
 */
public class Sigmoid implements IDifferentiableFunction {
   @Override
   public DoubleUnaryOperator fnc() {
      return z -> 1.0 / (1.0 + Math.exp(-z));
   }

   @Override
   public DoubleUnaryOperator derivative() {
      return y -> y * (1.0 - y);
   }
}
