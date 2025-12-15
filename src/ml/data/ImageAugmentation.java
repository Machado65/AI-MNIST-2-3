package ml.data;

import java.util.Random;

/**
 * Image augmentation techniques for improving model generalization.
 * Implements elastic deformation, Gaussian noise, rotation, and pixel shifts.
 * Based on techniques from: Best Practices for Convolutional Neural Networks
 * Applied to Visual Document Analysis (Simard et al., 2003).
 *
 * @author André Martins, António Matoso, Tomás Machado
 * @version 1.0
 * @see <a href=
 *      "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-31.pdf">Simard
 *      et al., 2003</a>
 */
public class ImageAugmentation {
   private static final int IMAGE_SIZE = 20; // MNIST 20x20
   private static final int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;

   private ImageAugmentation() {
      // Prevent instantiation
   }

   /**
    * Performs 1D convolution along a row or column.
    *
    * @param data       input image data
    * @param kernel     convolution kernel
    * @param radius     kernel radius
    * @param size       image dimension
    * @param x          x-coordinate
    * @param y          y-coordinate
    * @param horizontal true for horizontal, false for vertical
    * @return convolved value
    */
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

   /**
    * Builds a normalized Gaussian kernel for blurring.
    *
    * @param radius kernel radius
    * @param sigma  standard deviation of Gaussian
    * @return normalized Gaussian kernel
    */
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

   /**
    * Applies elastic deformation to an image.
    * Creates realistic distortions by generating random displacement fields
    * and smoothing them with Gaussian blur.
    *
    * @param input  input image (flattened 20x20)
    * @param output output buffer for deformed image
    * @param rand   random number generator
    * @param alpha  deformation intensity (typical: 6.0)
    * @param sigma  smoothing parameter (typical: 2.0)
    */
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

   /**
    * Shifts an image by random pixels in x and y directions.
    * Useful for position invariance during training.
    *
    * @param input    input image (flattened 20x20)
    * @param output   output buffer for shifted image
    * @param maxShift maximum shift in pixels (e.g., 2 means [-2, 2])
    * @param rand     random number generator
    */
   public static void shift(double[] input, double[] output,
         int maxShift, Random rand) {
      int shiftX = rand.nextInt(2 * maxShift + 1) - maxShift;
      int shiftY = rand.nextInt(2 * maxShift + 1) - maxShift;
      for (int y = 0; y < IMAGE_SIZE; ++y) {
         int srcY = y - shiftY;
         if (srcY < 0 || srcY >= IMAGE_SIZE) {
            continue;
         }
         for (int x = 0; x < IMAGE_SIZE; ++x) {
            int srcX = x - shiftX;
            if (srcX < 0 || srcX >= IMAGE_SIZE) {
               continue;
            }
            output[y * IMAGE_SIZE + x] = input[srcY * IMAGE_SIZE + srcX];
         }
      }
   }

   /**
    * Rotates an image by a random angle within specified bounds.
    * Uses backward mapping to avoid holes in the output.
    *
    * @param input      input image (flattened 20x20)
    * @param output     output buffer for rotated image
    * @param rand       random number generator
    * @param maxDegrees maximum rotation angle in degrees (e.g., 5.0 means [-5°,
    *                   +5°])
    */
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

   /**
    * Adjusts image brightness by a random factor.
    * Multiplies all pixels by a value in range [0.85, 1.15].
    *
    * @param input  input image (normalized to [0, 1])
    * @param output output buffer for brightness-adjusted image
    * @param rand   random number generator
    */
   public static void applyBrightness(double[] input, double[] output,
         Random rand) {
      double brightness = 0.85 + rand.nextDouble() * 0.3;
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         output[i] = Math.clamp(input[i] * brightness, 0.0,
               1.0);
      }
   }

   /**
    * Adds Gaussian noise to an image.
    * Each pixel gets independent random noise from N(0, stdDev²).
    *
    * @param input  input image (normalized to [0, 1])
    * @param output output buffer for noisy image
    * @param stdDev standard deviation of the noise (e.g., 0.02)
    * @param rand   random number generator
    * @return the output array (same as output parameter)
    */
   public static double[] gaussianNoise(double[] input, double[] output,
         double stdDev, Random rand) {
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         double noisy = input[i] + rand.nextGaussian() * stdDev;
         output[i] = Math.clamp(noisy, 0.0, 1.0);
      }
      return output;
   }

   /**
    * Applies random scaling (zoom in/out).
    *
    * @param input    input image
    * @param output   output buffer
    * @param rand     random generator
    * @param minScale minimum scale (e.g., 0.9)
    * @param maxScale maximum scale (e.g., 1.1)
    */
   public static void applyScaling(double[] input, double[] output,
         Random rand, double minScale, double maxScale) {
      double scale = minScale + rand.nextDouble() * (maxScale - minScale);
      double center = IMAGE_SIZE / 2.0;
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         output[i] = 0.0;
      }
      for (int y = 0; y < IMAGE_SIZE; ++y) {
         for (int x = 0; x < IMAGE_SIZE; ++x) {
            double srcX = (x - center) / scale + center;
            double srcY = (y - center) / scale + center;
            int ix = (int) Math.round(srcX);
            int iy = (int) Math.round(srcY);
            if (ix >= 0 && ix < IMAGE_SIZE && iy >= 0 && iy < IMAGE_SIZE) {
               output[y * IMAGE_SIZE + x] = input[iy * IMAGE_SIZE + ix];
            }
         }
      }
   }

   /**
    * Adjusts image contrast.
    *
    * @param input  input image [0,1]
    * @param output output buffer
    * @param rand   random generator
    * @param minC   minimum contrast (e.g., 0.8)
    * @param maxC   maximum contrast (e.g., 1.2)
    */
   public static void applyContrast(double[] input, double[] output,
         Random rand, double minC, double maxC) {
      double contrast = minC + rand.nextDouble() * (maxC - minC);
      for (int i = 0; i < IMAGE_PIXELS; ++i) {
         double v = 0.5 + contrast * (input[i] - 0.5);
         output[i] = Math.clamp(v, 0.0, 1.0);
      }
   }


}
