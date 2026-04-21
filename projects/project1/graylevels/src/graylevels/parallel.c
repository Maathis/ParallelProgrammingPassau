#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

#include "mpi.h"

static int np;    // number of MPI processes
static int self;  // rank of our MPI process

void compute_parallel(const struct TaskInput *TI) {
    double time_start, time_loaded, time_distributed;
    double time_levels, time_converted, time_collected;
    double time_saved;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single
            printf("Number of OMP threads in each MPI process: %d\n",
                   omp_get_num_threads());
        }
    }

    bool parallel_loading = TI->parallel_loading;

    MPI_Barrier(MPI_COMM_WORLD);
    time_start = seconds();

    if (parallel_loading) {
        // load image in a distributed fashion hiere
    } else if (self == 0) {
        // load image on processor 0
    }

    time_loaded = seconds();

    // distribute image (if needed)

    time_distributed = seconds();

    // determine new gray levels

    time_levels = seconds();

    // convert gray levels in image

    time_converted = seconds();

    // gather image

    time_collected = seconds();

    if (self == 0) {
        // save output image

        time_saved = seconds();
        printf("Times:\n"
               "  Loading:       %.6f s\n"
               "  Distributing:  %.6f s\n"
               "  Histogram:     %.6f s\n"
               "  Conversion:    %.6f s\n"
               "  Collecting:    %.6f s\n"
               "  Saving:        %.6f s\n"
               "  TOTAL:         %.6f s\n",
               time_loaded - time_start, time_distributed - time_loaded,
               time_levels - time_distributed, time_converted - time_levels,
               time_collected - time_converted, time_saved - time_collected,
               time_saved - time_start);
    }
}
