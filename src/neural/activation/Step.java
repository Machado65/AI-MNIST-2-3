package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511100822
 */
public class Step implements IDifferentiableFunction {
   private static double threshold = 0.5;

   @Override
   public DoubleUnaryOperator fnc() {
      return z -> (z < threshold) ? 0.0 : 1.0;
   }

   @Override
   public DoubleUnaryOperator derivative() {
      throw new UnsupportedOperationException(
            "Step function is not differentiable.");
   }

}
