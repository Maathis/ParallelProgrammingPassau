#include <stdbool.h>
#include <stdlib.h>
#include <sys/time.h>

struct TaskInput {
    // name of the input file
    char *filename;

    // name of the output file
    char *outfilename;

    // weight matrix: number of rows, number of columns
    // and entries (row-by-row)
    int weightsRows, weightsColumns;
    int *weights;

    // maximal number of smoother iterations to perform
    int maxIters;

    // stop smoothing when all pixel values (at inner pixels) change
    // by less than `minChange'.
    double minChange;

    // use median smoother instead of averaging smoother
    bool medianSmoother;

    // perform sobel operator after smoother
    bool doSobel;

    // use a two-dimensional data distribution
    bool twoDim;

    // number of MPI processes in the x- and y-dimension when a
    // two-dimensional data distributin is used (procsX * procsY
    // equals the total number of MPI processes)
    int procsX, procsY;

    // print some debug outputs during computation
    bool debugOutput;
};

void compute_single(const struct TaskInput *TI);
void compute_parallel(const struct TaskInput *TI);

// Returns the number of seconds since 1970-01-01T00:00:00.
inline static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000;
}
