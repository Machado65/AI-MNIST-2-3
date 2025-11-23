package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * Step (Heaviside) activation function implementation.
 * The step function outputs 0 for values below a threshold and 1 for values at
 * or above it.
 * This function is not differentiable and therefore cannot be used for
 * gradient-based learning.
 *
 * Function: step(z) = 0 if z < threshold, else 1
 * Derivative: Not defined (throws UnsupportedOperationException)
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511100822
 */
public class Step implements IDifferentiableFunction {
   private static double threshold = 0.5;

   /**
    * Returns the step activation function.
    * Outputs 0 if the input is less than the threshold, otherwise outputs 1.
    *
    * @return a DoubleUnaryOperator that applies the step function
    */
   @Override
   public DoubleUnaryOperator fnc() {
      return z -> (z < threshold) ? 0.0 : 1.0;
   }

   /**
    * The step function is not differentiable.
    * This method always throws an UnsupportedOperationException.
    *
    * @return never returns normally
    * @throws UnsupportedOperationException always, as the step function is not
    *                                       differentiable
    */
   @Override
   public DoubleUnaryOperator derivative() {
      throw new UnsupportedOperationException(
            "Step function is not differentiable.");
   }

}
