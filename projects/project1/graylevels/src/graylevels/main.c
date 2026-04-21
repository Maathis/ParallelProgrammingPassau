#include <getopt.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

#include "ppp/ppp.h"

static void usage(const char *progname) {
    int self;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    if (self == 0) {
        fprintf(stderr,
                "USAGE: %s -i input.pgm -o output.pgm [-n N] "
                "[-p] [-L]\n"
                "  input.pgm   a portable graymap image (grayscale)\n"
                "  output.pgm  a portable graymap image (grayscale)\n"
                "  N           number of gray levels in output, defaults to 2\n"
                "  -p          use parallel implementation (MPI and OpenMP)\n"
                "  -L          load image in parallel (only with -p)\n",
                progname);
    }
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }

    // Default parameter values
    struct TaskInput TI;
    TI.filename = NULL;
    TI.outfilename = NULL;
    TI.levels = 2;
    TI.parallel_loading = 0;
    bool parallel = false;

    int option;
    while ((option = getopt(argc, argv, "i:o:n:Lp")) != -1) {
        switch (option) {
        case 'i':
            TI.filename = strdup(optarg);
            break;
        case 'o':
            TI.outfilename = strdup(optarg);
            break;
        case 'n':
            TI.levels = atoi(optarg);
            break;
        case 'L':
            TI.parallel_loading = true;
            break;
        case 'p':
            parallel = true;
            break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (TI.filename == NULL) {
        fprintf(stderr, "Missing input file name\n");
        MPI_Finalize();
        return 1;
    }
    if (TI.outfilename == NULL) {
        fprintf(stderr, "Missing output file name\n");
        MPI_Finalize();
        return 1;
    }

    if (TI.levels < 2) {
        fprintf(stderr, "Number of levels must be at least 2\n");
        MPI_Finalize();
        return 1;
    }

    if (parallel) {
        compute_parallel(&TI);
    } else {
        compute_single(&TI);
    }

    free(TI.outfilename);
    free(TI.filename);

    MPI_Finalize();

    return 0;
}
