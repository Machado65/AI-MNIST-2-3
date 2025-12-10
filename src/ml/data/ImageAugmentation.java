package ml.data;

import java.util.Random;

/**
 * @link https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-31.pdf
 */
public class ImageAugmentation {
   private static final int IMAGE_SIZE = 20; // MNIST 20x20
   private static final int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;

   private ImageAugmentation() {
      // Prevent instantiation
   }

   private static double convolve1D(double[] data, double[] kernel,
         int radius, int size, int x, int y, boolean horizontal) {
      double acc = 0.0;
      for (int k = -radius; k <= radius; ++k) {
         int xx = horizontal ? x + k : x;
         int yy = horizontal ? y : y + k;
         if (xx >= 0 && xx < size && yy >= 0 && yy < size) {
            acc += data[yy * size + xx] * kernel[k + radius];
         }
      }
      return acc;
   }

   private static void verticalBlur(double[] temp, double[] out,
         double[] kernel, int radius, int size) {
      for (int y = 0; y < size; ++y) {
         for (int x = 0; x < size; ++x) {
            out[y * size + x] = convolve1D(temp, kernel, radius, size,
                  x, y, false);
         }
      }
   }

   private static double[] buildGaussianKernel(int radius, double sigma) {
      double[] kernel = new double[2 * radius + 1];
      double sum = 0.0;
      for (int i = -radius; i <= radius; ++i) {
         double v = Math.exp(-(i * i) / (2 * sigma * sigma));
         kernel[i + radius] = v;
         sum += v;
      }
      if (sum == 0.0) {
         sum = 1.0;
      }
      for (int i = 0; i < kernel.length; ++i) {
         kernel[i] /= sum;
      }
      return kernel;
   }

   private static void horizontalBlur(double[] input, double[] temp,
         double[] kernel, int radius, int size) {
      for (int y = 0; y < size; ++y) {
         for (int x = 0; x < size; ++x) {
            temp[y * size + x] = convolve1D(input, kernel, radius, size,
                  x, y, true);
         }
      }
   }

   private static double[] gaussianBlur(double[] input, double sigma) {
      int radius = (int) Math.ceil(3 * sigma);
      double[] kernel = buildGaussianKernel(radius, sigma);
      double[] temp = new double[IMAGE_PIXELS];
      horizontalBlur(input, temp, kernel, radius, IMAGE_SIZE);
      double[] out = new double[IMAGE_PIXELS];
      verticalBlur(temp, out, kernel, radius, IMAGE_SIZE);
      return out;
   }

   public static void elasticDeform(double[] input, double[] output,
         Random rand, double alpha, double sigma) {
      double[] dx = new double[IMAGE_PIXELS];
      double[] dy = new double[IMAGE_PIXELS];
      // Fill with random noise in [-1, 1]
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         dx[i] = (rand.nextDouble() * 2 - 1);
         dy[i] = (rand.nextDouble() * 2 - 1);
      }
      // Smooth noise with Gaussian kernel (simple separable approximation)
      double[] dxSmooth = gaussianBlur(dx, sigma);
      double[] dySmooth = gaussianBlur(dy, sigma);
      // Scale by alpha
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         dxSmooth[i] *= alpha;
         dySmooth[i] *= alpha;
      }
      // Apply displacement
      for (int y = 0; y < IMAGE_SIZE; ++y) {
         for (int x = 0; x < IMAGE_SIZE; ++x) {
            int idx = y * IMAGE_SIZE + x;
            double srcX = x + dxSmooth[idx];
            double srcY = y + dySmooth[idx];
            int ix = (int) Math.round(srcX);
            int iy = (int) Math.round(srcY);
            if (ix >= 0 && ix < IMAGE_SIZE && iy >= 0
                  && iy < IMAGE_SIZE) {
               output[idx] = input[iy * IMAGE_SIZE + ix];
            } else {
               output[idx] = 0.0;
            }
         }
      }
   }

   public static void shift(double[] input, double[] output,
         Random rand) {
      // Random shift: -2 to +2 pixels
      int shiftX = rand.nextInt(5) - 2;
      int shiftY = rand.nextInt(5) - 2;
      for (int y = 0; y < IMAGE_SIZE; ++y) {
         for (int x = 0; x < IMAGE_SIZE; ++x) {
            int srcX = x - shiftX;
            int srcY = y - shiftY;
            int dstIdx = y * IMAGE_SIZE + x;
            // Check bounds
            if (srcX >= 0 && srcX < IMAGE_SIZE && srcY >= 0
                  && srcY < IMAGE_SIZE) {
               int srcIdx = srcY * IMAGE_SIZE + srcX;
               output[dstIdx] = input[srcIdx];
            } else {
               output[dstIdx] = 0.0; // Black padding
            }
         }
      }
   }

   public static void applyRotation(double[] input, double[] output,
         Random rand, double maxDegrees) {
      // Random angle: -maxDegrees to +maxDegrees
      double angleDeg = (rand.nextDouble() * 2.0 - 1.0) * maxDegrees;
      double angleRad = angleDeg * Math.PI / 180.0;
      double centerX = IMAGE_SIZE / 2.0;
      double centerY = IMAGE_SIZE / 2.0;
      // Use inverse rotation for backward mapping
      double cos = Math.cos(-angleRad);
      double sin = Math.sin(-angleRad);
      for (int i = 0; i < output.length; ++i) {
         output[i] = 0.0;
      }
      for (int y = 0; y < IMAGE_SIZE; ++y) {
         for (int x = 0; x < IMAGE_SIZE; ++x) {
            // Map output pixel back to input space
            double srcX = cos * (x - centerX) - sin * (y - centerY) + centerX;
            double srcY = sin * (x - centerX) + cos * (y - centerY) + centerY;
            int ix = (int) Math.round(srcX);
            int iy = (int) Math.round(srcY);
            if (ix >= 0 && ix < IMAGE_SIZE && iy >= 0
                  && iy < IMAGE_SIZE) {
               output[y * IMAGE_SIZE + x] = input[iy * IMAGE_SIZE + ix];
            }
         }
      }
   }

   public static void applyBrightness(double[] input, double[] output,
         Random rand) {
      double brightness = 0.85 + rand.nextDouble() * 0.3;
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         output[i] = Math.clamp(input[i] * brightness, 0.0,
               1.0);
      }
   }

   public static double[] gaussianNoise(double[] input, double[] output,
         double stdDev, Random rand) {
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         double noisy = input[i] + rand.nextGaussian() * stdDev;
         output[i] = Math.clamp(noisy, 0.0, 1.0);
      }
      return output;
   }
}
