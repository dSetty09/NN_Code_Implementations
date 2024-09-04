#ifndef CONVO_LAYER_H
#define CONVO_LAYER_H

#include "../common_definitions.h"

static const unsigned int GRAY = 0;
static const unsigned int RGB = 1;


/** KERNEL DEFINITIONS **/


float convolution(float* kernel_mat, float* sup_mat, int nrows, int ncols) {
    float result = 0;

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            result += mat_val(kernel_mat, ncols, i, j) * mat_val(sup_mat, ncols, i, j);
        }
    }

    return result;
}

typedef struct kernel_node {
    float* matrix;
    int nrows;
    int ncols;

    float (*convo_func) (float*, float*, int, int);
} Kernel;


/** 3D KERNEL DEFINITIONS **/

float convolution3d(Kernel* kernels, int nkerns, float** mat3d) {
    float result = 0;

    for (int k = 0; k < nkerns; ++k) {
        result += kernels[k].convo_func(kernels[k].matrix, mat3d[k], kernels[k].nrows, kernels[k].ncols);
    }

    return result;
}

typedef struct kernel_node_3d {
    Kernel* channel_kerns;
    int num_channels;

    float (*convo_3d) (Kernel*, int, float**);
} Kernel3D;


/** CONVOLUTIONAL LAYER DEFINITIONS **/

void convl_builder(Kernel** kernels_ref, int* num_kernels_ref, int nkerns, int kern_nrows, int kern_ncols, float* kern_content) {

}

float* convl_exec(float** img, int img_height, int img_len, int img_type, Kernel* kernels, int num_kernels, int padding, int stride) {
    int width = 1; // width is synonymous with the number of channels
    if (img_type == RGB) width = 3;

    return *img;
}

typedef struct convo_layer {
    Kernel* kernels;
    int num_kernels;

    void (*build) (Kernel**, int*, int, int, int, float*);
    float* (*exec) (float**, int, int, int, Kernel*, int, int, int);
} ConvoLayer;

#endif