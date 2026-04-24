#include <math.h>
#include <stdio.h>
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

static int imageSize, imageSizePerProcess;
uint8_t *recimage;

void debug(int *hist, int size) {
    for (int i = 0; i < size; ++i) {
        printf("[rank : %d | hist : %d] > %d\n",self,i,hist[i]);
    }
}

static void compute_levels(int levels, uint8_t *newlevels, uint8_t *recimg) {
    // histogram[x] counts the number of pixels that have a
    // gray level value of x.
    const int HISTOGRAM_SIZE = maxcolor + 1;
    int histogram[HISTOGRAM_SIZE];

    // initialize histogram entries to 0
    for (int i = 0; i <= maxcolor; ++i)
        histogram[i] = 0;

    // compute the histogram
    for (int i = 0; i < imageSizePerProcess; ++i)
        histogram[recimg[i]]++;

    const int BUFFER_SIZE = np*HISTOGRAM_SIZE;
    int buffer[BUFFER_SIZE];
    MPI_Allgather(histogram, HISTOGRAM_SIZE, MPI_INT, buffer, HISTOGRAM_SIZE, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < HISTOGRAM_SIZE; ++i)
    {
        histogram[i] = 0;
    }

    for (int i = 0; i < BUFFER_SIZE; ++i)
    {
        histogram[i%HISTOGRAM_SIZE] += buffer[i];
    }

    // debug(histogram, HISTOGRAM_SIZE);

    // histsum[x] is the number of pixels with gray values
    // that are less than or equal to x.
    int histsum[HISTOGRAM_SIZE];
    histsum[0] = histogram[0];
    for (int i = 1; i <= maxcolor; ++i)
        histsum[i] = histogram[i] + histsum[i - 1];


    // 'step' is the number of pixels mapped to the
    // same new gray value.
    int step = (imageSize + levels) / levels;
    for (int i = 0; i <= maxcolor; ++i) {
        newlevels[i] = ((histsum[i] / step) * maxcolor) / (levels - 1);
    }
}

static void convert_grayvalues(uint8_t *recimage, uint8_t *newlevels) {
    // set new gray values in the image
    for (int i = 0; i < imageSizePerProcess; ++i) {
        recimage[i] = newlevels[recimage[i]];
    }
}

// partfn is a pointer to a function which is called with kind, rows and columns of the image; it is expected to set values for *offset and *length which describe the part of the image to be loaded (measured in pixels). partfn must return a pointer to buffer where the image part is to be stored.
// calloc or malloc must be used to allocate the buffer.
// ppp_pnm_read_part returns the pointer returned by partfn.
uint8_t *partfn(enum pnm_kind kind, int rows, int columns, int *offset, int *length)
{
    if (kind != PNM_KIND_PGM)
        return NULL;

    imageSize = rows*columns;
    imageSizePerProcess = imageSize/np;
    
    *offset = imageSizePerProcess*self;
    
    if(self == (np-1) && imageSize%np != 0) {
        imageSizePerProcess += imageSize%np;
    }
    *length = imageSizePerProcess;
    uint8_t *buffer = (uint8_t*) malloc(imageSizePerProcess*sizeof(uint8_t));

    return buffer;
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
        recimage = ppp_pnm_read_part(TI->filename, 
            &kind, 
            &rows, 
            &columns, 
            &maxcolor, 
            partfn
        );
        
        if(self == 0) {
            image = (uint8_t*) malloc(imageSize*sizeof(uint8_t));
        }
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
    int countsSend[np];
    int displacements[np];
    if(!parallel_loading)
    {
        int config[3] = {columns, rows, maxcolor};
        MPI_Bcast(&config, 3, MPI_INT, 0, MPI_COMM_WORLD);
        if(self != 0)
        {
            columns = config[0];
            rows = config[1];
            maxcolor = config[2];
        }

        printf("%d > columns : %d | rows : %d | maxcolor : %d\n", self, columns, rows, maxcolor);
        
        imageSize = (rows*columns);
        imageSizePerProcess = imageSize/np; // Case limit : IMAGE_SIZE%np != 0

        for(int i = 0; i < np; ++i)
        {
            countsSend[i] = imageSizePerProcess;
            displacements[i] = i*imageSizePerProcess;
        }

        int remainder = imageSize%np;
        if(remainder != 0) {
            countsSend[np-1] += remainder;
        }
            
        if(self == (np-1) && remainder != 0)
        {
            imageSizePerProcess += remainder;
        }
        recimage = (uint8_t*) malloc(imageSizePerProcess*sizeof(uint8_t));

        MPI_Scatterv(image, countsSend, displacements, MPI_UINT8_T, recimage,countsSend[self], MPI_UINT8_T,0,MPI_COMM_WORLD);
    } else { // Remainder for parallel loading
        for(int i = 0; i < np; ++i)
        {
            countsSend[i] = imageSizePerProcess;
            displacements[i] = i*imageSizePerProcess;
        }

        int remainder = imageSize%np;
        if(remainder != 0) {
            countsSend[np-1] += remainder;
        }
    }
    time_distributed = seconds();
    
    // newlevels[x] is the new gray value a pixel with original
    // gray value x should get.
    uint8_t newlevels[maxcolor + 1];

    // determine new gray levels
    compute_levels(TI->levels, newlevels, recimage);
    time_levels = seconds();

    // convert gray levels in image
    convert_grayvalues(recimage, newlevels);
    time_converted = seconds();

    // gather image
    MPI_Gatherv(
        recimage,imageSizePerProcess,MPI_UINT8_T,image,countsSend,displacements,MPI_UINT8_T,0,MPI_COMM_WORLD
    );
        
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
        free(recimage);
    }
}
