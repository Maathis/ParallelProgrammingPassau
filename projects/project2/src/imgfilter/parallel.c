#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

void compute_parallel(const struct TaskInput *TI) {
    int self, np;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single
            printf("Number of OMP threads in each MPI process: %d\n", omp_get_num_threads());
        }
    }

    // Load image here

    double time_loaded = seconds();

    // Process image here according to the options in TI

    double time_computed = seconds();

    // Collect final image data in process 0 here.

    if (self == 0) {
        if (TI->outfilename != NULL) {
            // Write output image here
        }

        printf("Computation time: %.6f\n", time_computed - time_loaded);
    }
}
