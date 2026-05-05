#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppp/ppp.h"

static void usage(const char *progname) {
    int self;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    if (self == 0) {
        fprintf(stderr,
                "USAGE: %s -i input.pgm [-o output.pgm] [-n] [-s] [-m]\n"
                "    [-N n] [-c mc] [-M weight_matrix] [-2]"
                "    [-d] [-h]\n"
                "  n      max repetitions of smoother (default: 40)\n"
                "  mc     smooth until change is below mc (default 0.2)\n"
                "  WM     matrix of smoothing weights,\n"
                "         defaults to 1,2,1/2,3,2/1,2,1\n"
                "  -n     use naive sequential implementation\n"
                "  -s     run Sobel operator after smoothing\n"
                "  -m     use median smoother instead of average smoother\n"
                "  -2     use a 2d distribution\n"
                "  -d     give some debug/progress output\n"
                "  -h     print this help\n",
                progname);
    }
}

/*
 * Parse weight matrix in syntax from `str'
 *   1,2,1/2,3,2/1,2,1
 * (values in a row are separated by ',' and rows are separated by '/')
 * and return the number of rows and columns (in `rows' and 'cols', respectively)
 * and a pointer to the matrix. Return NULL on format error.
 */
static int *parseMatrix(const char *str, int *rows, int *cols) {
    const char *p;
    char *next;
    int count;

    *rows = 1;
    count = 1;
    p = str;
    do {
        strtol(p, &next, 10);
        if (next != NULL) {
            p = next + 1;
            if (*next == ',') {
                count++;
            } else if (*next == '/') {
                (*rows)++;
                count++;
            } else if (*next == '\0')
                break;
            else
                return NULL;
        } else
            return NULL;
    } while (true);
    if (count % *rows != 0)
        return NULL;
    *cols = count / *rows;

    int *weights = (int *)malloc(*rows * *cols * sizeof(double));
    if (weights == NULL)
        return NULL;

    p = str;
    for (int i = 0; i < *rows * *cols; ++i) {
        weights[i] = strtol(p, &next, 10);
        p = next + 1;
    }
    return weights;
}

static int weights_default[] = {1, 2, 1, 2, 3, 2, 1, 2, 1};

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }

    int self, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Default parameter values
    struct TaskInput TI;
    TI.filename = NULL;
    TI.outfilename = NULL;

    TI.weights = weights_default;
    TI.weightsRows = 3;
    TI.weightsColumns = 3;
    TI.minChange = 0.2;
    TI.maxIters = 20;

    TI.medianSmoother = false;
    TI.doSobel = false;
    TI.twoDim = false;
    TI.debugOutput = false;

    bool naive = false;

    int option;
    while ((option = getopt(argc, argv, "i:o:N:c:M:snm2dh")) != -1) {
        switch (option) {
        case 'i':
            TI.filename = strdup(optarg);
            break;
        case 'o':
            TI.outfilename = strdup(optarg);
            break;
        case 'N':
            TI.maxIters = atoi(optarg);
            break;
        case 'c':
            TI.minChange = atof(optarg);
            break;
        case 'M':
            TI.weights = parseMatrix(optarg, &(TI.weightsRows), &(TI.weightsColumns));
            break;
        case 's':
            TI.doSobel = true;
            break;
        case 'n':
            naive = true;
            break;
        case 'm':
            TI.medianSmoother = true;
            break;
        case '2':
            TI.twoDim = true;
            break;
        case 'd':
            TI.debugOutput = true;
            break;
        case 'h':
            usage(argv[0]);
            MPI_Finalize();
            return 0;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (TI.weights == NULL) {
        if (self == 0)
            fprintf(stderr, "Could not parse weights\n");
        MPI_Finalize();
        return 1;
    }

    if (TI.weightsColumns % 2 == 0 || TI.weightsRows % 2 == 0) {
        if (self == 0)
            fprintf(stderr, "Weight matrix must have odd number of rows and columns\n");
        MPI_Finalize();
        return 1;
    }

    if (TI.filename == NULL || TI.outfilename == NULL) {
        if (self == 0)
            usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (TI.medianSmoother) {
        int sumWeights = 0;
        for (int r = 0; r < TI.weightsRows; ++r) {
            for (int c = 0; c < TI.weightsColumns; ++c) {
                int w = TI.weights[TI.weightsColumns * r + c];
                if (w < 0) {
                    if (self == 0)
                        fprintf(stderr, "Weights must be non-negative for median smoother\n");
                    MPI_Finalize();
                    return 1;
                }
                sumWeights += w;
            }
        }
        if (sumWeights % 2 == 0) {
            if (self == 0)
                fprintf(stderr, "Sum of weights must be odd for median smoother\n");
            MPI_Finalize();
            return 1;
        }
    }

    if (TI.debugOutput && self == 0) {
        printf("Weight matrix:\n");
        for (int r = 0; r < TI.weightsRows; ++r) {
            for (int c = 0; c < TI.weightsColumns; ++c) {
                printf(" %4d", TI.weights[TI.weightsColumns * r + c]);
            }
            printf("\n");
        }
    }

    if (TI.twoDim) {
        // Compute TI.procsX and T.procsY such that TI.procsX * TI.procsY == np
        // and 1 <= TI.procsX <= TI.procsY and TI.procsX is as large as possible.
        for (int i = 1; i * i <= np; ++i) {
            if (np % i == 0) {
                TI.procsX = i;
            }
        }
        TI.procsY = np / TI.procsX;

        if (TI.debugOutput && self == 0) {
            printf("procsX=%d, procsY=%d\n", TI.procsX, TI.procsY);
        }
    }

    if (naive)
        compute_single(&TI);
    else
        compute_parallel(&TI);

    free(TI.outfilename);
    free(TI.filename);

    MPI_Finalize();

    return 0;
}
