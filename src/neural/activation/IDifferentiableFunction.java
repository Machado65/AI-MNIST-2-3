package neural.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511101224
 */
public interface IDifferentiableFunction {
   DoubleUnaryOperator fnc();

   DoubleUnaryOperator derivative();
}
