package neural.activation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.DoubleUnaryOperator;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Step Activation Function Tests")
class StepTest {

   private final Step step = new Step();

   @Nested
   @DisplayName("Function Tests")
   class FunctionTests {

      @Test
      @DisplayName("Should return 0 for values below threshold")
      void testStepBelowThreshold() {
         DoubleUnaryOperator fnc = step.fnc();
         assertEquals(0.0, fnc.applyAsDouble(0.0));
         assertEquals(0.0, fnc.applyAsDouble(0.4));
         assertEquals(0.0, fnc.applyAsDouble(-1.0));
      }

      @Test
      @DisplayName("Should return 1 for values at or above threshold")
      void testStepAboveThreshold() {
         DoubleUnaryOperator fnc = step.fnc();
         assertEquals(1.0, fnc.applyAsDouble(0.5));
         assertEquals(1.0, fnc.applyAsDouble(1.0));
         assertEquals(1.0, fnc.applyAsDouble(10.0));
      }

      @Test
      @DisplayName("Should return binary values only")
      void testStepBinary() {
         DoubleUnaryOperator fnc = step.fnc();
         for (double x = -10.0; x <= 10.0; x += 0.1) {
            double result = fnc.applyAsDouble(x);
            assertTrue(result == 0.0 || result == 1.0);
         }
      }
   }

   @Nested
   @DisplayName("Derivative Tests")
   class DerivativeTests {

      @Test
      @DisplayName("Should throw UnsupportedOperationException")
      void testDerivativeThrowsException() {
         assertThrows(UnsupportedOperationException.class,
               () -> step.derivative().applyAsDouble(0.5));
      }

      @Test
      @DisplayName("Should throw exception with correct message")
      void testDerivativeExceptionMessage() {
         UnsupportedOperationException exception = assertThrows(UnsupportedOperationException.class,
               () -> step.derivative().applyAsDouble(0.5));
         assertTrue(exception.getMessage().contains("not differentiable"));
      }
   }

   @Nested
   @DisplayName("IDifferentiableFunction Interface Tests")
   class InterfaceTests {

      @Test
      @DisplayName("Should implement IDifferentiableFunction")
      void testImplementsInterface() {
         assertTrue(step instanceof IDifferentiableFunction);
      }

      @Test
      @DisplayName("Should return non-null function")
      void testFncNotNull() {
         assertNotNull(step.fnc());
      }

      @Test
      @DisplayName("Derivative should throw UnsupportedOperationException")
      void testDerivativeThrows() {
         assertThrows(UnsupportedOperationException.class,
               () -> new Step().derivative());
      }
   }
}
