package neural;

import math.Matrix;
import neural.activation.IDifferentiableFunction;

/**
 * @author hdaniel@ualg.pt
 * @author Tomás Machado
 * @version 202511052038
 */
public class MLP {
   private Matrix[] w; // weights for each layer
   private Matrix[] b; // biases for each layer
   private Matrix[] yp; // outputs for each layer (y predicted)
   private final IDifferentiableFunction[] act; // activation functions for each layer
   private final int nLayers;
   private final int nLayers1;

   /*
    * PRE: layerSizes.length >= 2
    * PRE: act.length == layerSizes.length - 1
    */
   public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
      if (seed < 0) {
         seed = (int) System.currentTimeMillis();
      }
      this.nLayers = layerSizes.length;
      this.nLayers1 = this.nLayers - 1;
      // setup activation by layer
      this.act = act;
      // create output storage for each layer but the input layer
      this.yp = new Matrix[this.nLayers];
      // create weights and biases for each layer
      // each row in w[l] represents the weights that are input
      this.w = new Matrix[this.nLayers1];
      this.b = new Matrix[this.nLayers1];
      for (int i = 0; i < this.nLayers1; ++i) {
         this.w[i] = Matrix.rand(layerSizes[i], layerSizes[i + 1], seed);
         this.b[i] = Matrix.rand(1, layerSizes[i + 1], seed);
      }
   }

   // Feed forward propagation
   // also used to predict after training the net
   // yp[0] = X
   // yp[l+1] = Sigmoid(yp[l] * w[l]+b[l])
   public Matrix predict(Matrix x) {
      this.yp[0] = x;
      for (int l = 0; l < this.nLayers1; ++l) {
         yp[l + 1] = yp[l].dot(w[l]).addRowVector(b[l])
               .apply(act[l].fnc());
      }
      return this.yp[this.nLayers - 1];
   }

   private void updateLayer(int l, int l1, Matrix delta, Matrix e,
         double lr) {
      // delta = e .* yp[l+1] .* (1-yp[l+1])
      delta.set(e.mult(this.yp[l1].apply(this.act[l].derivative())));
      // w[l] += yp[l]^T * delta * lr
      this.w[l] = this.w[l].add(
            this.yp[l].transpose().dot(delta).mult(lr));
      // b[l] += sum(delta) * lr
      b[l] = b[l].addRowVector(delta.sumColumns().mult(lr));
   }

   // public Matrix backPropagation(Matrix y, double lr) {
   // Matrix e = null;
   // Matrix delta = null;// dummy initialization
   // // back propagation using generalized delta rule
   // int n = this.nLayers - 2;
   // int n1 = n + 1;
   // // e = y - yp[l+1](l == n)
   // e = y.sub(this.yp[n1]);
   // this.updateLayer(n, n1, delta, e, lr);
   // for (int l = n - 1; l >= 0; --l) {
   // int l1 = l + 1;
   // // e = delta * w[l+1]^T
   // e = delta.dot(this.w[l1].transpose());
   // this.updateLayer(l, l1, delta, e, lr);
   // }

   // return e;
   // }

   public Matrix backPropagation(Matrix y, double lr) {
      Matrix e = null;
      Matrix delta = null;
      // back propagation using generalized delta rule
      int n = this.nLayers - 2;
      for (int l = n; l >= 0; --l) {
         int l1 = l + 1;
         e = (l == n)
               // e = y - yp[l+1](l == n)
               ? y.sub(this.yp[l1])
               // else e = delta * w[l+1]^T
               : delta.dot(this.w[l1].transpose());
         // delta = e .* yp[l+1] * (1-yp[l+1])
         delta = e.mult(this.yp[l1].apply(this.act[l].derivative()));
         // w[l] += yp[l]^T * delta * lr
         this.w[l] = this.w[l].add(
               this.yp[l].transpose().dot(delta).mult(lr));
         // b[] += sum(delta) * lr
         b[l] = b[l].addRowVector(delta.sumColumns().mult(lr));
      }
      return e;
   }

   public double[] train(Matrix x, Matrix y, double learningRate,
         int epochs) {
      int nSamples = x.rows();
      double[] mse = new double[epochs];
      for (int epoch = 0; epoch < epochs; ++epoch) {
         // forward propagation
         predict(x);
         // backward propagation
         Matrix e = backPropagation(y, learningRate);
         // mse
         mse[epoch] = e.dot(e.transpose()).get(0, 0)
               / nSamples;
      }
      return mse;
   }
}
