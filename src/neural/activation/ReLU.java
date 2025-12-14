package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * Rectified Linear Unit (ReLU) activation function.
 * One of the most popular activation functions in deep learning.
 * Simple and computationally efficient, but can suffer from "dying ReLU"
 * problem.
 *
 * Function: f(x) = max(0, x)
 * Derivative: f'(x) = 1 if x > 0, else 0
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class ReLU implements IDifferentiableFunction {
   /**
    * Returns the ReLU activation function.
    *
    * @return function that applies ReLU: f(x) = max(0, x)
    */
   @Override
   public DoubleUnaryOperator fnc() {
      return x -> (x > 0) ? x : 0;
   }

   /**
    * Returns the derivative of the ReLU function.
    * The derivative is 1 for positive inputs and 0 otherwise.
    *
    * @return function computing f'(x) = 1 if x > 0, else 0
    */
   @Override
   public DoubleUnaryOperator derivative() {
      return x -> (x > 0.0) ? 1.0 : 0.0;
   }
}
