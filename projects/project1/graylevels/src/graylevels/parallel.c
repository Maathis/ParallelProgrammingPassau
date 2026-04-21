#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

#include "mpi.h"

static int np;    // number of MPI processes
static int self;  // rank of our MPI process

static int rows, columns;
static int maxcolor;

static void compute_levels(int levels, uint8_t *newlevels, int sharedLen, uint8_t *recimg) {
    // histogram[x] counts the number of pixels that have a
    // gray level value of x.
    int histogram[maxcolor + 1];

    // initialize histogram entries to 0
    for (int i = 0; i <= maxcolor; ++i)
        histogram[i] = 0;

    // compute the histogram
    for (int i = 0; i < sharedLen; ++i)
        histogram[recimg[i]]++;

    // histsum[x] is the number of pixels with gray values
    // that are less than or equal to x.
    int histsum[maxcolor + 1];
    histsum[0] = histogram[0];
    for (int i = 1; i <= maxcolor; ++i)
        histsum[i] = histogram[i] + histsum[i - 1];

    // 'step' is the number of pixels mapped to the
    // same new gray value.
    int step = (sharedLen + levels) / levels;
    for (int i = 0; i <= maxcolor; ++i) {
        newlevels[i] = ((histsum[i] / step) * maxcolor) / (levels - 1);
    }
}

static void convert_grayvalues(uint8_t *recimage,  int sharedLen, uint8_t *newlevels) {
    // set new gray values in the image
    for (int i = 0; i < sharedLen; ++i) {
        recimage[i] = newlevels[recimage[i]];
    }
}

void compute_parallel(const struct TaskInput *TI) {

    uint8_t *image;

    double time_start, time_loaded, time_distributed;
    double time_levels, time_converted, time_collected;
    double time_saved;

    enum pnm_kind kind;


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
        image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxcolor);
    
        if (image == NULL) {
            fprintf(stderr, "Could not load image from file '%s'.\n",
                TI->filename);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }

        if (kind != PNM_KIND_PGM || maxcolor < 1) {
            fprintf(stderr, "Image '%s' is not a portable graymap.\n",
                    TI->filename);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }

        if (TI->levels > maxcolor+1) {
            fprintf(stderr, "Gray levels requested (%d) exceed image format (%d)\n",
                    TI->levels, maxcolor+1);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }

    }

    time_loaded = seconds();

    // distribute image (if needed)
    int config[3] = {columns, rows, maxcolor};
    MPI_Bcast(&config, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if(self != 0)
    {
        columns = config[0];
        rows = config[1];
        maxcolor = config[2];
    }

    printf("%d > columns : %d | rows : %d | maxcolor : %d\n", self, columns, rows, maxcolor);
    
    const int IMAGE_SIZE = (rows*columns);
    const int IMAGE_SIZE_PER_PROCESS = IMAGE_SIZE/np; // TODO : fix if np%2 == 1
    uint8_t recimage[IMAGE_SIZE_PER_PROCESS];

    printf("%d > %d image size fully / %d image size per process\n",self, IMAGE_SIZE, IMAGE_SIZE_PER_PROCESS);
    MPI_Scatter(image, 
        IMAGE_SIZE_PER_PROCESS, 
        MPI_UINT8_T,
        recimage, 
        IMAGE_SIZE_PER_PROCESS,
        MPI_UINT8_T, 
        0, 
        MPI_COMM_WORLD);

    time_distributed = seconds();

    // newlevels[x] is the new gray value a pixel with original
    // gray value x should get.
    uint8_t newlevels[maxcolor + 1];

    // determine new gray levels
    printf("Level = %d | newlevels = %d\n", TI->levels, maxcolor+1);
    compute_levels(TI->levels, newlevels, IMAGE_SIZE_PER_PROCESS, recimage);
    time_levels = seconds();

    // convert gray levels in image
    convert_grayvalues(recimage, IMAGE_SIZE_PER_PROCESS, newlevels);
    time_converted = seconds();

    // gather image
    MPI_Gather(recimage, 
        IMAGE_SIZE_PER_PROCESS, 
        MPI_UINT8_T, 
        image, 
        IMAGE_SIZE_PER_PROCESS, 
        MPI_UINT8_T, 
        0, MPI_COMM_WORLD);
    time_collected = seconds();

    if (self == 0) {
        // save output image
        if (ppp_pnm_write(TI->outfilename, PNM_KIND_PGM, rows, columns, maxcolor, image) != 0)
            fprintf(stderr, "Could not write image to file '%s'.\n", TI->outfilename);
        
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
        free(image);
    }
}
