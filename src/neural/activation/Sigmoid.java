package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * Sigmoid activation function implementation.
 * The sigmoid function maps any real value to the range (0, 1),
 * making it useful for binary classification and as a smooth activation
 * function.
 *
 * Function: σ(z) = 1 / (1 + e^(-z))
 * Derivative: σ'(y) = y * (1 - y), where y is the output of the sigmoid
 * function
 *
 * @author hdaniel@ualg.pt
 * @author André Martins, António Matoso, Tomás Machado
 * @version 202511101225
 */
public class Sigmoid implements IDifferentiableFunction {
   /**
    * Returns the sigmoid activation function.
    * Computes σ(z) = 1 / (1 + e^(-z))
    *
    * @return a DoubleUnaryOperator that applies the sigmoid function
    */
   @Override
   public DoubleUnaryOperator fnc() {
      return z -> 1.0 / (1.0 + Math.exp(-z));
   }

   /**
    * Returns the derivative of the sigmoid function.
    * For numerical stability, the derivative is computed as y * (1 - y),
    * where y is the output of the sigmoid function (not the input z).
    *
    * @return a DoubleUnaryOperator that computes the derivative
    */
   @Override
   public DoubleUnaryOperator derivative() {
      return y -> y * (1.0 - y);
   }
}
