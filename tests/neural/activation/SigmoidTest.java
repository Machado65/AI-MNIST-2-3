package neural.activation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.DoubleUnaryOperator;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Sigmoid Activation Function Tests")
class SigmoidTest {

   private final Sigmoid sigmoid = new Sigmoid();

   @Nested
   @DisplayName("Function Tests")
   class FunctionTests {

      @Test
      @DisplayName("Should return value between 0 and 1")
      void testSigmoidRange() {
         DoubleUnaryOperator fnc = sigmoid.fnc();
         assertTrue(fnc.applyAsDouble(0.0) > 0.0 && fnc.applyAsDouble(0.0) < 1.0);
         assertTrue(fnc.applyAsDouble(5.0) > 0.0 && fnc.applyAsDouble(5.0) < 1.0);
         assertTrue(fnc.applyAsDouble(-5.0) > 0.0 && fnc.applyAsDouble(-5.0) < 1.0);
      }

      @Test
      @DisplayName("Should return 0.5 for input 0")
      void testSigmoidZero() {
         DoubleUnaryOperator fnc = sigmoid.fnc();
         assertEquals(0.5, fnc.applyAsDouble(0.0), 0.0001);
      }

      @Test
      @DisplayName("Should approach 1 for large positive values")
      void testSigmoidLargePositive() {
         DoubleUnaryOperator fnc = sigmoid.fnc();
         assertTrue(fnc.applyAsDouble(10.0) > 0.99);
      }

      @Test
      @DisplayName("Should approach 0 for large negative values")
      void testSigmoidLargeNegative() {
         DoubleUnaryOperator fnc = sigmoid.fnc();
         assertTrue(fnc.applyAsDouble(-10.0) < 0.01);
      }

      @Test
      @DisplayName("Should be symmetric around 0.5")
      void testSigmoidSymmetry() {
         DoubleUnaryOperator fnc = sigmoid.fnc();
         double pos = fnc.applyAsDouble(2.0);
         double neg = fnc.applyAsDouble(-2.0);
         assertEquals(1.0, pos + neg, 0.0001);
      }
   }

   @Nested
   @DisplayName("Derivative Tests")
   class DerivativeTests {

      @Test
      @DisplayName("Should return derivative value between 0 and 1")
      void testDerivativeRange() {
         DoubleUnaryOperator deriv = sigmoid.derivative();
         assertTrue(deriv.applyAsDouble(0.5) >= 0.0 && deriv.applyAsDouble(0.5) <= 1.0);
      }

      @Test
      @DisplayName("Should return maximum derivative at y=0.5")
      void testDerivativeMaximum() {
         DoubleUnaryOperator deriv = sigmoid.derivative();
         double maxDeriv = deriv.applyAsDouble(0.5);
         assertEquals(0.25, maxDeriv, 0.0001);
      }

      @Test
      @DisplayName("Should return 0 at y=0")
      void testDerivativeAtZero() {
         DoubleUnaryOperator deriv = sigmoid.derivative();
         assertEquals(0.0, deriv.applyAsDouble(0.0), 0.0001);
      }

      @Test
      @DisplayName("Should return 0 at y=1")
      void testDerivativeAtOne() {
         DoubleUnaryOperator deriv = sigmoid.derivative();
         assertEquals(0.0, deriv.applyAsDouble(1.0), 0.0001);
      }

      @Test
      @DisplayName("Should be symmetric around 0.5")
      void testDerivativeSymmetry() {
         DoubleUnaryOperator deriv = sigmoid.derivative();
         double at03 = deriv.applyAsDouble(0.3);
         double at07 = deriv.applyAsDouble(0.7);
         assertEquals(at03, at07, 0.0001);
      }
   }

   @Nested
   @DisplayName("IDifferentiableFunction Interface Tests")
   class InterfaceTests {

      @Test
      @DisplayName("Should implement IDifferentiableFunction")
      void testImplementsInterface() {
         assertTrue(sigmoid instanceof IDifferentiableFunction);
      }

      @Test
      @DisplayName("Should return non-null function")
      void testFncNotNull() {
         assertNotNull(sigmoid.fnc());
      }

      @Test
      @DisplayName("Should return non-null derivative")
      void testDerivativeNotNull() {
         assertNotNull(sigmoid.derivative());
      }
   }
}
