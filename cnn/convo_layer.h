#ifndef CONVO_LAYER_H
#define CONVO_LAYER_H

#include "../common_definitions.h"

static const unsigned int GRAY = 0;
static const unsigned int RGB = 1;

static const unsigned int NO_PADDING = 0;
static const unsigned int PADDING = 1;


/** KERNEL DEFINITIONS **/

/*
 * For each cell in a kernel matrix, calculates the product of multiplying that cell value by the value of the cell, 
 * in the superimposed section of the image, that shares the same row and column index, with respect to the 
 * superimposed section, as the cell in the kernel matrix. Then, sums all the calculated products.
 * 
 * @param kernmat | The kernel matrix
 * @param kern_dimen | The dimension of the length and height of a kernel matrix
 * @param img | The image the kernel is being passed over
 * @param img_ncols | The number of columns in the image the kernel is passing over
 * @param sup_centrow | The center row index of the section of the image superimposed by the kernel matrix
 * @param sup_centcol | The center column index of the section of the image superimposed by the kernel matrix
 * 
 * @return The summation of each product of each respective cell between the kernel matrix and superimposed section of the image.
 */
float sup_product_summation(float* kern_mat, float* img, int kern_dimen, int img_ncols, int sup_cent_row, int sup_cent_col) {
    float result = 0;

    for (int kern_i = 0, sup_i = sup_cent_row - (kern_dimen / 2); kern_i < kern_dimen; ++kern_i, ++sup_i) {
        for (int kern_j = 0, sup_j = sup_cent_col - (kern_dimen / 2); kern_j < kern_dimen; ++kern_j, ++sup_j) {
            result += mat_val(kern_mat, kern_dimen, kern_i, kern_j) * mat_val(img, img_ncols, sup_i, sup_j);
        }
    }

    return result;
}

/*
 * A two-dimensional kernel used for convolution of a two-dimensional image.
 */
typedef struct kernel_node {
    float* matrix; // The kernel matrix
    int dimen; // The row and column dimension of the kernel matrixs
    float (*sup_prod_sum) (float*, float*, int, int, int, int); // The superimposed product summation function for this Kernel
} Kernel;


/** 3D KERNEL DEFINITIONS **/

/*
 * For each channel, calculates the superimposed product summation between the kernel and superimposed section of the image 
 * associated with the bespoke channel. Then, sums each superimposed product summation.
 * 
 * @param channel_kerns | A list of kernels for each respective channel
 * @param img3d | The 3D image of which each channel is being superimposed by the kernel sharing their channel at the same time
 * @param img3d_ncols | The number of columns in the 3D image
 * @param sup_cent_row | The center row index of the section of the 3D image being superimposed across all channels
 * @param sup_cent_col | The center column index of the section of the 3D image being superimposed across all channels
 * @param nchannels | The number of channels in the 3D image
 * 
 * @return The summation of each superimposed product summation of each channel.
 */
float sup_product_summation_3d(Kernel* channel_kerns, float** img3d, int img3d_ncols, int sup_cent_row, int sup_cent_col, int nchannels) {
    float result = 0;

    for (int ch = 0; ch < nchannels; ++ch) {
        result += channel_kerns[ch].sup_prod_sum(channel_kerns[ch].matrix, img3d[ch], channel_kerns[ch].dimen, img3d_ncols, sup_cent_row, sup_cent_col);
    }

    return result;
}

/*
 * A three-dimensional kernel used for convolution of a 3D image.
 */
typedef struct kernel_node_3d {
    Kernel* channel_kerns; // list of kernels for each channel being convolved over
    int num_channels; // the number of channels being convolved over

    float (*sup_prod_sum_3d) (Kernel*, float**, int, int, int, int); // The 3D superimposed product summation function for this 3D kernel
} Kernel3D;


/** CONVOLUTIONAL LAYER DEFINITIONS **/

/* void convl_builder(Kernel** kernels_ref, Kernel3D** kernels3d_ref, int* num_kernels_ref, int* num_channels_ref, 
                   int nkerns, int nchannels, int kern_nrows, int kern_ncols, float* kern_content) {

    *num_kernels_ref = nkerns;
    *num_channels_ref = nchannels;

    // Allocate for referenced kernel list, if only 1 channel; otherwise, allocate for referenced 3d kernel list
    if (nchannels == 1) { 
        *kernels_ref = (Kernel**) malloc(sizeof(Kernel*) * nkerns);
        *kernels3d_ref = NULL;
    } else {
        *kernels_ref = NULL;
        *kernels3d_ref = (Kernel3D**) malloc(sizeof(Kernel3D*) * nkerns);
    }

    for (int k = 0; k < nkerns; ++k) {
        // Allocate memory for list of kernels, where number of entries is equal to number of channels
        Kernel* channel_kernels = (Kernel*) malloc(sizeof(Kernel) * nchannels);

        for (int ch = 0; ch < nchannels; ++ch) {
            // Allocate memory for the matrix of the kernel associated with the current channel
            float* kmatrix = (float*) malloc(sizeof(float) * kern_nrows * kern_ncols);

            // Initialize values for the kernel matrix
            for (int i = 0; i < kern_nrows; ++i) {
                for (int j = 0; j < kern_ncols; ++j) {
                    if (kern_content) {
                        set_mat_val(kmatrix, kern_ncols, i, j, mat_val(kern_content, kern_ncols, i, j));
                    } else {
                        set_mat_val(kmatrix, kern_ncols, i, j, random_num(-2, 2, 3));
                    }
                }
            }

            // Initialize kernel for current channel
            channel_kernels[ch].matrix = kmatrix;
            channel_kernels[ch].nrows = kern_nrows;
            channel_kernels[ch].ncols = kern_ncols;
            channel_kernels[ch].convo_func = convolution;
        }

        // Assign index in referenced kernels list or kernels 3d list to the newly created 2d or 3d kernel
        if (nchannels == 1) {
            (*kernels_ref)[k] = channel_kernels[0];
        } else {
            (*kernels3d_ref)[k].channel_kerns = channel_kernels;
            (*kernels3d_ref)[k].num_channels = nchannels;
            (*kernels3d_ref)[k].convo_3d = convolution3d;
        }
    }
}

float* convl_exec(Kernel* kernels, Kernel3D* kernels3d, int num_kernels, int num_channels, float** img, int img_nrows, int img_ncols, int padding, int stride) {
    int kern_nrows;
    int kern_ncols;

    if (num_channels == 1) {
        kern_nrows = kernels->nrows;
        kern_ncols = kernels->ncols;
    } else {
        kern_nrows = kernels3d->channel_kerns->nrows;
        kern_ncols = kernels3d->channel_kerns->ncols;
    }

    int result_img_dimen = img_nrows;

    if (!padding) {
        result_img_dimen = (img_nrows - kern_nrows) / stride + 1;
    }

    int result_img_dimen = (img_nrows - kern_nrows) / stride + 1;

    // Allocate memory for result image
    float* result_img = (float*) malloc(sizeof(float) * result_img_dimen);

    for (int k = 0; k < num_kernels; ++k) {
        // (r,c) represent image coordinates while (i, j) represent result image coordinates
        for (int r = kern_nrows / 2, i = 0; r <= img_nrows - kern_nrows / 2; r += stride) {
            for (int c = kern_ncols / 2, j = 0; c <= img_ncols - kern_ncols / 2; c += stride) {
                if (num_channels == 1) {
                    float* sup_mat = (float*) malloc(sizeof(float) * kern_nrows * kern_ncols);
                    for (int sup_mat_i = r - kern_nrows / 2; sup_mat_i <= r + kern_nrows / 2; ++sup_mat_i) {
                        for (int sup_mat_j = c - kern_ncols / 2; sup_mat_j <= c + kern_ncols / 2; ++sup_mat_j) {
                            // continue blah blah blah
                        }
                    }

                    set_mat_val(result_img, result_img_dimen, i, j, kernels[k].convo_func(kernels[k].matrix, ))
                }
            }
        }
    }
}

typedef struct convo_layer {
    Kernel* kernels;
    Kernel3D* kernels3d;

    int num_kernels;
    int num_channels;

    void (*build) (Kernel**, Kernel3D**, int*, int*, int, int, int, float*);
    float* (*exec) (float**, int, int, int, Kernel*, int, int, int);
} ConvoLayer; */

#endif