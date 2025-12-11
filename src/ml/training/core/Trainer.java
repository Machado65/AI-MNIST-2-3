package ml.training.core;

import java.util.Random;

import math.Matrix;
import ml.training.config.TrainConfig;
import ml.training.result.EvaluationMetrics;
import ml.training.result.EvaluationResult;
import ml.training.result.PredictionStats;
import ml.training.result.TrainResult;
import ml.training.threshold.OptimalThreshold;
import neural.MLP;
import neural.activation.IDifferentiableFunction;

public class Trainer {
   TrainConfig config;
   private MLP mlp;

   public Trainer(int[] topology, IDifferentiableFunction[] act,
         TrainConfig config, Random rand) {
      this.config = new TrainConfig(config);
      this.mlp = new MLP(topology, act, rand);
   }

   public MLP getMLP() {
      return this.mlp;
   }

   public TrainResult train() {
      return mlp.train(config);
   }

   public Matrix predict(Matrix x) {
      return mlp.predict(x);
   }

   private int countCorrectPredictions(Matrix pred, Matrix actual,
         double threshold, int n) {
      int ans = 0;
      for (int i = 0; i < n; ++i) {
         if (((pred.get(i, 0) < threshold)
               ? 0
               : 1) == (int) actual.get(i, 0)) {
            ++ans;
         }
      }
      return ans;
   }

   private OptimalThreshold findOptimalThreshold(Matrix pred,
         Matrix actual, int n) {
      double bestThreshold = 0.5;
      double bestAccuracy = 0.0;
      for (double threshold = 0.1; threshold <= 0.90; threshold += 0.05) {
         double accuracy = (double) countCorrectPredictions(
               pred, actual, threshold, n) / n;
         if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestThreshold = threshold;
         }
      }
      return new OptimalThreshold(bestThreshold);
   }

   private EvaluationMetrics computeMetrics(Matrix pred, Matrix actual,
         double threshold, int n) {
      EvaluationMetrics metrics = new EvaluationMetrics();
      for (int i = 0; i < n; ++i) {
         metrics.update((pred.get(i, 0) < threshold) ? 0 : 1,
               (int) actual.get(i, 0));
      }
      return metrics;
   }

   public EvaluationResult evaluate() {
      Matrix teX = config.getTe().getX();
      Matrix teY = config.getTe().getY();
      Matrix pred = predict(teX);
      int n = teX.rows();
      OptimalThreshold optimalThreshold = findOptimalThreshold(pred,
            teY, n);
      return new EvaluationResult(new PredictionStats(pred),
            optimalThreshold, computeMetrics(pred, teY,
                  optimalThreshold.getThreshold(), n));
   }
}
