#include <math.h>
#include <stdio.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

/*
 * Convert a gray value in the range 0..maxcolor to a double value in [0,1].
 */
inline static double grayvalueToDouble(uint8_t v, int maxcolor) {
    return (double)v / maxcolor;
}

/*
 * Convert a double value back to a gray value in the range 0..maxcolor.
 */
inline static int grayvalueFromDouble(double d, int maxcolor) {
    int v = (int)lrint(d * maxcolor);
    return (v < 0 ? 0 : (v > maxcolor ? maxcolor : v));
}

/*
 * Swap the values pointed to by 'a' and 'b',
 * i.e., swap the two double pointers that 'a' and 'b'
 * point to.
 */
static void swap(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

/*
 * Comparison function for `qsort'. Return -1 if
 * the first argument is less than the second argument,
 * 1 if the second argument is less and 0 otherwise.
 */
static int cmpdouble(const void *a_, const void *b_) {
    double a = *((const double *)a_);
    double b = *((const double *)b_);
    if (a < b)
        return -1;
    else if (a > b)
        return 1;
    return 0;
}

/*
 * Apply smoother to *input. Use *temp as a temporary storage. `image' and `temp' are swapped
 * after each iteration, i.e., when the function returns, the result is again in `image'.
 */
static void smooth(double **input, double **temp, int rows, const int columns,
                   const struct TaskInput *TI) {
    // Use 0 as pixel value if the pixel is outside the bounds of the image.
#define S(c, r)                                                                                    \
    ((r) >= 0 && (r) < rows && (c) >= 0 && (c) < columns ? (*input)[(r) * columns + (c)] : 0)

    const int wRows = TI->weightsRows;
    const int wCols = TI->weightsColumns;
    const int nWeights = wRows * wCols;

    // Sum up the weights in the weight matrix.
    int sumWeights = 0;
    for (int i = 0; i < nWeights; ++i)
        sumWeights += TI->weights[i];

    // The averaging smoother divides by the absolute value
    // of sum of the values in the weight matrix (if this sum is not 0).
    double avgFactor = sumWeights != 0 ? abs(sumWeights) : 1;

    // Perform (at most) maxIters iterations.
    for (int iter = 0; iter < TI->maxIters; ++iter) {
        double maxchange = 0; // maximal change in pixel value at an inner pixel
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < columns; ++x) {
                double new_value;
                if (TI->medianSmoother) {
                    // Each neighbor value of (x,y) is put as many
                    // times in `data' as the corresponding weight in
                    // the weight matrix says.
                    double data[sumWeights];
                    int idx = 0;
                    int w = 0;
                    for (int v = -wRows / 2; v <= wRows / 2; ++v) {
                        for (int u = -wCols / 2; u <= wCols / 2; ++u) {
                            for (int i = 0; i < TI->weights[idx]; ++i)
                                data[w++] = S(x + u, y + v);
                            idx++;
                        }
                    }
                    // The new pixel value is the median (the value in the middle
                    // after sorting the values).
                    qsort(data, sumWeights, sizeof(double), cmpdouble);
                    new_value = data[sumWeights / 2];
                } else {
                    // The averaging filter computes the weighted average
                    // of the neighboring pixels.
                    int idx = 0;
                    double sum = 0;
                    for (int v = -wRows / 2; v <= wRows / 2; ++v) {
                        for (int u = -wCols / 2; u <= wCols / 2; ++u) {
                            sum += TI->weights[idx] * S(x + u, y + v);
                            ++idx;
                        }
                    }
                    new_value = sum / avgFactor;
                }

                // Store new value.
                (*temp)[y * columns + x] = new_value;

                // Update maxchange (only at pixel not at the border).
                double change = fabs(new_value - S(x, y));
                if (change > maxchange && x > 0 && y > 0 && x < columns - 1 && y < rows - 1)
                    maxchange = change;
            }
        }

        // What is now in *temp is used as input in the next iteration,
        // i.e., we swap `input' and `temp'.
        swap(input, temp);

        if (TI->debugOutput)
            printf("Iteration %d: maxchange=%g\n", iter, maxchange);

        // When `maxchange' drops below a threshold, no further iterations are performed.
        if (maxchange < TI->minChange)
            break;
    }
#undef S
}

/*
 * Apply the Sobel operator to *input using *temp as output.
 * `input' and `temp' are swapped, i.e., when the function returns,
 * the result is in *input.
 */
static void sobel(double **input, double **temp, int rows, const int columns) {
#define S(c, r)                                                                                    \
    ((r) >= 0 && (r) < rows && (c) >= 0 && (c) < columns ? (*input)[(r) * columns + (c)] : 0)

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
            double sx, sy;
            sx = S(x - 1, y - 1) + 2 * S(x, y - 1) + S(x + 1, y - 1) //
                 - S(x - 1, y + 1) - 2 * S(x, y + 1) - S(x + 1, y + 1);
            sy = S(x - 1, y - 1) + 2 * S(x - 1, y) + S(x - 1, y + 1) //
                 - S(x + 1, y - 1) - 2 * S(x + 1, y) - S(x + 1, y + 1);
            (*temp)[y * columns + x] = hypot(sx, sy);
        }
    }

    swap(input, temp);
#undef S
}

void compute_single(const struct TaskInput *TI) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image;

    image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxcolor);

    if (image == NULL) {
        fprintf(stderr, "Could not load image from file '%s'.\n", TI->filename);
        exit(1);
    } else if (kind != PNM_KIND_PGM) {
        fprintf(stderr, "Image is not a \"portable graymap.\"\n");
        exit(1);
    }

    // Filter operations are performed in double values
    // (in the range [0,1]) and processing is from one array (which is read)
    // to another array (which is written to).
    double *imageD = (double *)malloc(sizeof(double) * rows * columns);
    double *tempD = (double *)malloc(sizeof(double) * rows * columns);
    if (imageD == NULL || tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for the image\n");
        exit(1);
    }

    // Copy the original `image' to `imageD' and convert it to double values.
    for (int i = 0; i < rows * columns; ++i)
        imageD[i] = grayvalueToDouble(image[i], maxcolor);

    double time_loaded = seconds();

    // Filter operations swap the pointers `imageD' and `tempD' after each
    // computation step; therefore, we must pass addresses to these
    // variables. The result of the computation is always in `imageD'.

    // Perform smoother first, then edge detection (if enabled).
    smooth(&imageD, &tempD, rows, columns, TI);
    if (TI->doSobel)
        sobel(&imageD, &tempD, rows, columns);

    double time_computed = seconds();

    // Copy the result from `imageD' back to `image', converting the double
    // values to integers again.
    for (int i = 0; i < rows * columns; ++i)
        image[i] = grayvalueFromDouble(imageD[i], maxcolor);

    free(imageD);
    free(tempD);

    if (TI->outfilename != NULL) {
        if (ppp_pnm_write(TI->outfilename, kind, rows, columns, maxcolor, image) == -1) {
            fprintf(stderr, "Could not write output to '%s'.\n", TI->outfilename);
        }
    }

    printf("Computation time: %.6f\n", time_computed - time_loaded);
}
