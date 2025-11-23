package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * Interface for differentiable activation functions used in neural networks.
 * Provides both the function itself and its derivative for backpropagation.
 *
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511101224
 */
public interface IDifferentiableFunction {
   /**
    * Returns the activation function.
    *
    * @return a DoubleUnaryOperator representing the activation function
    */
   DoubleUnaryOperator fnc();

   /**
    * Returns the derivative of the activation function.
    * Used during backpropagation for gradient computation.
    *
    * @return a DoubleUnaryOperator representing the derivative of the activation
    *         function
    */
   DoubleUnaryOperator derivative();
}
