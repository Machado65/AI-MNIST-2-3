package ml.training;

public class EvaluationResult {
   public final PredictionStats p;
   public final OptimalThreshold o;
   public final EvaluationMetrics m;

   public EvaluationResult(PredictionStats p, OptimalThreshold o,
         EvaluationMetrics m) {
      this.p = p;
      this.o = o;
      this.m = m;
   }

   public OptimalThreshold getOptimalThreshold() {
      return this.o;
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(this.p).append("\n");
      sb.append(this.o).append("\n");
      sb.append(this.m);
      return sb.toString();
   }
}
