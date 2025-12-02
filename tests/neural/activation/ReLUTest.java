package neural.activation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.DoubleUnaryOperator;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("ReLU Activation Function Tests")
class ReLUTest {

   private final ReLU relu = new ReLU();

   @Nested
   @DisplayName("Function Tests")
   class FunctionTests {

      @Test
      @DisplayName("Should return 0 for negative input")
      void testReLUNegative() {
         DoubleUnaryOperator fnc = relu.fnc();
         assertEquals(0.0, fnc.applyAsDouble(-5.0));
         assertEquals(0.0, fnc.applyAsDouble(-0.1));
      }

      @Test
      @DisplayName("Should return input for positive values")
      void testReLUPositive() {
         DoubleUnaryOperator fnc = relu.fnc();
         assertEquals(5.0, fnc.applyAsDouble(5.0));
         assertEquals(0.1, fnc.applyAsDouble(0.1));
         assertEquals(100.0, fnc.applyAsDouble(100.0));
      }

      @Test
      @DisplayName("Should return 0 for input 0")
      void testReLUZero() {
         DoubleUnaryOperator fnc = relu.fnc();
         assertEquals(0.0, fnc.applyAsDouble(0.0));
      }

      @Test
      @DisplayName("Should be non-negative")
      void testReLUNonNegative() {
         DoubleUnaryOperator fnc = relu.fnc();
         assertTrue(fnc.applyAsDouble(-100.0) >= 0.0);
         assertTrue(fnc.applyAsDouble(0.0) >= 0.0);
         assertTrue(fnc.applyAsDouble(100.0) >= 0.0);
      }
   }

   @Nested
   @DisplayName("Derivative Tests")
   class DerivativeTests {

      @Test
      @DisplayName("Should return 0 for non-positive input")
      void testDerivativeNonPositive() {
         DoubleUnaryOperator deriv = relu.derivative();
         assertEquals(0.0, deriv.applyAsDouble(0.0));
         assertEquals(0.0, deriv.applyAsDouble(-5.0));
         assertEquals(0.0, deriv.applyAsDouble(-0.1));
      }

      @Test
      @DisplayName("Should return 1 for positive input")
      void testDerivativePositive() {
         DoubleUnaryOperator deriv = relu.derivative();
         assertEquals(1.0, deriv.applyAsDouble(5.0));
         assertEquals(1.0, deriv.applyAsDouble(0.1));
         assertEquals(1.0, deriv.applyAsDouble(100.0));
      }
   }

   @Nested
   @DisplayName("IDifferentiableFunction Interface Tests")
   class InterfaceTests {

      @Test
      @DisplayName("Should implement IDifferentiableFunction")
      void testImplementsInterface() {
         assertTrue(relu instanceof IDifferentiableFunction);
      }

      @Test
      @DisplayName("Should return non-null function")
      void testFncNotNull() {
         assertNotNull(relu.fnc());
      }

      @Test
      @DisplayName("Should return non-null derivative")
      void testDerivativeNotNull() {
         assertNotNull(relu.derivative());
      }
   }
}
