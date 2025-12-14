package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * Leaky Rectified Linear Unit (Leaky ReLU) activation function.
 * A variant of ReLU that allows small negative values instead of zero.
 * This helps prevent "dying ReLU" problem where neurons can become inactive.
 *
 * Function: f(x) = x if x > 0, else α*x (where α = 0.01)
 * Derivative: f'(x) = 1 if x > 0, else α
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 */
public class LeakyReLU implements IDifferentiableFunction {
   private static final double ALPHA = 0.01;

   /**
    * Returns the Leaky ReLU activation function.
    *
    * @return function that applies Leaky ReLU: f(x) = max(x, 0.01*x)
    */
   @Override
   public DoubleUnaryOperator fnc() {
      return x -> (x > 0) ? x : ALPHA * x;
   }

   /**
    * Returns the derivative of the Leaky ReLU function.
    * The derivative is 1 for positive inputs and 0.01 for negative inputs.
    *
    * @return function computing f'(x) = 1 if x > 0, else 0.01
    */
   @Override

   public DoubleUnaryOperator derivative() {
      return x -> (x > 0) ? 1.0 : ALPHA;
   }
}
