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
static int imageSize, imageSizePerProcess; // imageSizePerProcess is the local size
static int maxcolor;

uint8_t *localImage;

void debug(int *arr, int size) {
    for (int i = 0; i < size;++i) {
        printf("[rank : %d | hist : %d] > %d\n", self,i,arr[i]);
    }
}


// Question C
static void compute_levels(int levels, uint8_t *newlevels) {

    const int HISTOGRAM_SIZE = maxcolor + 1;
    // histogram[x] counts the number of pixels that have a
    // gray level value of x.
    int distributedHistogram[HISTOGRAM_SIZE];
    // initialize histogram entries to 0
    for (int i = 0; i <= maxcolor; ++i)
        distributedHistogram[i] = 0;
  

    #pragma omp parallel for reduction(+:distributedHistogram[:])
    for (int i = 0; i < imageSizePerProcess; ++i) {
        distributedHistogram[localImage[i]]++;
    }


    int buffer[np*HISTOGRAM_SIZE];
    MPI_Allgather(distributedHistogram, HISTOGRAM_SIZE, MPI_INT, buffer, HISTOGRAM_SIZE, MPI_INT, MPI_COMM_WORLD);

    // Calculate the full histogram after getting all parts
    int histogram[HISTOGRAM_SIZE];
    for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
        histogram[i] = 0;
    }

    #pragma omp parallel for reduction(+:histogram[:])
    for(int i = 0; i < (np*HISTOGRAM_SIZE); ++i) {
        histogram[i%HISTOGRAM_SIZE] += buffer[i];
    }

    // Calculate histsum and newlevels
    int histsum[HISTOGRAM_SIZE];
    histsum[0] = histogram[0];
    
    // That seems to be non-parallelizable because histsum[i] depends on histsum[i-1]
    for (int i = 1; i <= maxcolor; ++i)
        histsum[i] = histogram[i] + histsum[i - 1];

    // 'step' is the number of pixels mapped to the
    // same new gray value.
    int step = (imageSize + levels) / levels;

    #pragma omp parallel for
    for (int i = 0; i <= maxcolor; ++i) {
        newlevels[i] = ((histsum[i] / step) * maxcolor) / (levels - 1);
    }
}

// Question D
static void compute_levels_master(int levels, uint8_t *newlevels) {

    const int HISTOGRAM_SIZE = maxcolor + 1;
    // histogram[x] counts the number of pixels that have a
    // gray level value of x.
    int distributedHistogram[HISTOGRAM_SIZE];
    // initialize histogram entries to 0
    for (int i = 0; i <= maxcolor; ++i)
        distributedHistogram[i] = 0;
  

    // Calculate the histogram on the local image for the question d (master only)
    int threadsHistogram[omp_get_max_threads()][HISTOGRAM_SIZE];

    #pragma omp parallel
    {
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            threadsHistogram[omp_get_thread_num()][i] = 0;
        }

        #pragma omp for
        for (int i = 0; i < imageSizePerProcess; ++i) {
            threadsHistogram[omp_get_thread_num()][localImage[i]]++;
        }

        #pragma omp critical
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            distributedHistogram[i] += threadsHistogram[omp_get_thread_num()][i];
        }
    }


    int buffer[np*HISTOGRAM_SIZE];
    MPI_Allgather(distributedHistogram, HISTOGRAM_SIZE, MPI_INT, buffer, HISTOGRAM_SIZE, MPI_INT, MPI_COMM_WORLD);

    // Calculate the full histogram after getting all parts
    int histogram[HISTOGRAM_SIZE];
    for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
        histogram[i] = 0;
    }

    int threadsBuffer[omp_get_max_threads()][HISTOGRAM_SIZE];

    #pragma omp parallel
    {
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            threadsBuffer[omp_get_thread_num()][i] = 0;
        }

        #pragma omp for
        for(int i = 0; i < (np*HISTOGRAM_SIZE); ++i) {
            threadsBuffer[omp_get_thread_num()][i%HISTOGRAM_SIZE] += buffer[i];
        }

        #pragma omp critical
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            histogram[i] += threadsBuffer[omp_get_thread_num()][i];
        }
    }

    // Calculate histsum and newlevels
    int histsum[HISTOGRAM_SIZE];
    histsum[0] = histogram[0];
    
    // That seems to be non-parallelizable because histsum[i] depends on histsum[i-1]
    for (int i = 1; i <= maxcolor; ++i)
        histsum[i] = histogram[i] + histsum[i - 1];

    // 'step' is the number of pixels mapped to the
    // same new gray value.
    int step = (imageSize + levels) / levels;

    #pragma omp parallel for
    for (int i = 0; i <= maxcolor; ++i) {
        newlevels[i] = ((histsum[i] / step) * maxcolor) / (levels - 1);
    }
}

static void convert_grayvalues(uint8_t *newlevels) {
    // set new gray values in the image
    #pragma omp parallel for
    for (int i = 0; i < imageSizePerProcess; ++i) {
        localImage[i] = newlevels[localImage[i]];
    }
}

/*
 * Load a part of an image on the current processor
 */
uint8_t *partfn(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
    if (kind != PNM_KIND_PGM)
        return NULL;

    imageSize = rows*columns;
    imageSizePerProcess = imageSize/np;
    
    *offset = imageSizePerProcess*self;

    if(self == (np-1) && imageSize%np != 0) {
        imageSizePerProcess += imageSize%np; // The last process gets the remainder 
    }
    *length = imageSizePerProcess;

    /*
     * Allocate space for the image part.
     * On processor 0 we allocate space for the whole
     * result image.
     */
    return (uint8_t *)malloc(imageSizePerProcess*sizeof(uint8_t));
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
        localImage = ppp_pnm_read_part(TI->filename, &kind, &rows, &columns, &maxcolor, partfn);

        if(self == 0) {
            image = (uint8_t*) malloc(imageSize*sizeof(uint8_t));
        }

    } else if (self == 0) {
        // load image on processor 0
        image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxcolor);

        if (image == NULL) {
            fprintf(stderr, "Could not load image from file '%s'.\n",
                    TI->filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }

        if (kind != PNM_KIND_PGM || maxcolor < 1) {
            fprintf(stderr, "Image '%s' is not a portable graymap.\n",
                    TI->filename);
            free(localImage);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }

        if (TI->levels > maxcolor+1) {
            fprintf(stderr, "Gray levels requested (%d) exceed image format (%d)\n",
                    TI->levels, maxcolor+1);
            free(localImage);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }
    }

    time_loaded = seconds();

    // distribute image (if needed)
    int countsSent[np], displacements[np];
    if(parallel_loading) {
        for(int i = 0; i < np; ++i) {
            countsSent[i] = imageSizePerProcess;
            displacements[i] = i*imageSizePerProcess;
        }

        // Fix the edge case
        int remainder = imageSize%np;
        if(remainder != 0) {
            countsSent[np-1] += remainder; // The last process gets the rest of the image
        }
    } else {
        int config[3] = {columns, rows, maxcolor};
        MPI_Bcast(&config, 3, MPI_INT, 0, MPI_COMM_WORLD);
        if(self != 0) {
            columns = config[0];
            rows = config[1];
            maxcolor = config[2];
        }

        imageSize = rows*columns;
        imageSizePerProcess = imageSize/np; // edge case imageSize%np != 0

        for(int i = 0; i < np; ++i) {
            countsSent[i] = imageSizePerProcess;
            displacements[i] = i*imageSizePerProcess;
        }

        // Fix the edge case
        int remainder = imageSize%np;
        if(remainder != 0) {
            countsSent[np-1] += remainder; // The last process gets the rest of the image
        }

        if(self == (np-1) && remainder != 0) { // We update the local size of the last process
            imageSizePerProcess += remainder;
        }

        localImage = (uint8_t*) malloc(imageSizePerProcess*sizeof(uint8_t));
        MPI_Scatterv(image, countsSent, displacements, MPI_UINT8_T, localImage, countsSent[self], MPI_UINT8_T, 0, MPI_COMM_WORLD);
    }

    time_distributed = seconds();

    // determine new gray levels
    uint8_t newlevels[maxcolor + 1];

    // compute_levels(TI->levels, newlevels);
    compute_levels_master(TI->levels, newlevels);

    time_levels = seconds();

    // convert gray levels in image
    convert_grayvalues(newlevels);

    time_converted = seconds();

    // gather image
    MPI_Gatherv(localImage, imageSizePerProcess, MPI_UINT8_T, image, countsSent, displacements, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    time_collected = seconds();

    if (self == 0) {
        // save output image
        if(ppp_pnm_write(TI->outfilename, PNM_KIND_PGM, rows, columns, maxcolor, image) != 0)
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
        free(localImage);
    }
}
