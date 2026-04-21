#include "stdio.h"
#include "sys/time.h"

struct TaskInput {
    // filename: name of the input file
    char *filename;

    // outfilename: name of the output file
    char *outfilename;

    // number of gray levels in output
    int levels;

    // parallel_loading: enable parallel image loading
    int parallel_loading;
};

// Converts an image (sequential implementation).
void compute_single(const struct TaskInput *TI);

// Converts an image using OpenMP and MPI.
void compute_parallel(const struct TaskInput *TI);

// Returns the number of seconds since 1970-01-01T00:00:00.
inline static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000;
}
