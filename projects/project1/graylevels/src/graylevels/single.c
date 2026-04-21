#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"


/* The image parameters: height, width and maximal value per color channel.
 */
static int rows, columns;
static int maxcolor;

/* Pointer to the input image.
 */
static uint8_t *image;

static void compute_levels(int levels, uint8_t *newlevels) {
    const int len = rows * columns;

    // histogram[x] counts the number of pixels that have a
    // gray level value of x.
    int histogram[maxcolor + 1];

    // initialize histogram entries to 0
    for (int i = 0; i <= maxcolor; ++i)
        histogram[i] = 0;

    // compute the histogram
    for (int i = 0; i < len; ++i)
        histogram[image[i]]++;

    // histsum[x] is the number of pixels with gray values
    // that are less than or equal to x.
    int histsum[maxcolor + 1];
    histsum[0] = histogram[0];
    for (int i = 1; i <= maxcolor; ++i)
        histsum[i] = histogram[i] + histsum[i - 1];

    // 'step' is the number of pixels mapped to the
    // same new gray value.
    int step = (len + levels) / levels;
    for (int i = 0; i <= maxcolor; ++i) {
        newlevels[i] = ((histsum[i] / step) * maxcolor) / (levels - 1);
    }
}

static void convert_grayvalues(uint8_t *newlevels) {
    const int len = rows * columns;

    // set new gray values in the image
    for (int i = 0; i < len; ++i) {
        image[i] = newlevels[image[i]];
    }
}

void compute_single(const struct TaskInput *TI) {
    enum pnm_kind kind;
    double time_start, time_loaded;
    double time_levels, time_converted;
    double time_saved;

    time_start = seconds();

    image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxcolor);

    if (image == NULL) {
        fprintf(stderr, "Could not load image from file '%s'.\n",
                TI->filename);
        exit(1);
    }

    if (kind != PNM_KIND_PGM || maxcolor < 1) {
        fprintf(stderr, "Image '%s' is not a portable graymap.\n",
                TI->filename);
        free(image);
        exit(1);
    }

    if (TI->levels > maxcolor+1) {
        fprintf(stderr, "Gray levels requested (%d) exceed image format (%d)\n",
                TI->levels, maxcolor+1);
        free(image);
        exit(1);
    }

    time_loaded = seconds();

    // newlevels[x] is the new gray value a pixel with original
    // gray value x should get.
    uint8_t newlevels[maxcolor + 1];

    // determine new gray levels
    compute_levels(TI->levels, newlevels);
    time_levels = seconds();

    // convert gray levels in image
    convert_grayvalues(newlevels);
    time_converted = seconds();

    if (ppp_pnm_write(TI->outfilename, PNM_KIND_PGM, rows, columns, maxcolor, image) != 0)
        fprintf(stderr, "Could not write image to file '%s'.\n", TI->outfilename);
    time_saved = seconds();

    printf("Times:\n"
           "  Loading:       %.6f s\n"
           "  Histogram:     %.6f s\n"
           "  Conversion:    %.6f s\n"
           "  Saving:        %.6f s\n"
           "  TOTAL:         %.6f s\n",
           time_loaded - time_start, time_levels - time_loaded,
           time_converted - time_levels, time_saved - time_converted,
           time_saved - time_start);

    free(image);
}
