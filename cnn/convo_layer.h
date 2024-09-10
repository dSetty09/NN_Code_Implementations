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
 * -> NOTE: THIS FUNCTION'S BEHAVIOR IS UNDEFINED IF THE KERNEL IS LARGER THAN THE IMAGE IN ANY DIMENSION
 * 
 * @param kernmat | The kernel matrix
 * @param kern_nrows | The number of rows in the kernel matrix
 * @param kern_ncols | The number of columns in the kernel matrix
 * @param img | The image the kernel is being passed over
 * @param img_ncols | The number of columns in the image the kernel is passing over
 * @param sup_start_row | The starting row index of the section of the image superimposed by the kernel matrix
 * @param sup_start_col | The starting column index of the section of the image superimposed by the kernel matrix
 * 
 * @return The summation of each product of each respective cell between the kernel matrix and superimposed section of the image.
 */
float sup_product_summation(float* kern_mat, float* img, int kern_nrows, int kern_ncols, int img_ncols, int sup_start_row, int sup_start_col) {
    float result = 0;

    for (int kern_i = 0, sup_i = sup_start_row; kern_i < kern_nrows; ++kern_i, ++sup_i) {
        for (int kern_j = 0, sup_j = sup_start_col; kern_j < kern_ncols; ++kern_j, ++sup_j) {
            float ni = mat_val(kern_mat, kern_ncols, kern_i, kern_j);
            float nj = mat_val(img, img_ncols, sup_i, sup_j);
            result += mat_val(kern_mat, kern_ncols, kern_i, kern_j) * mat_val(img, img_ncols, sup_i, sup_j);
        }
    }

    return result;
}

/*
 * A two-dimensional kernel used for convolution of a two-dimensional image.
 */
typedef struct kernel_node {
    float* matrix; // The kernel matrix
    int nrows; // The number of rows in this kernel
    int ncols; // Then umber of columns in this kernel
    float (*sup_prod_sum) (float*, float*, int, int, int, int, int); // The superimposed product summation function for this Kernel
} Kernel;


/** 3D KERNEL DEFINITIONS **/

/*
 * For each channel, calculates the superimposed product summation between the kernel and superimposed section of the image 
 * associated with the bespoke channel. Then, sums each superimposed product summation.
 * 
 * @param channel_kerns | A list of kernels for each respective channel
 * @param img3d | The 3D image of which each channel is being superimposed by the kernel sharing their channel at the same time
 * @param img3d_ncols | The number of columns in the 3D image
 * @param sup_start_row | The starting row index of the section of the 3D image being superimposed across all channels
 * @param sup_start_col | The starting column index of the section of the 3D image being superimposed across all channels
 * @param nchannels | The number of channels in the 3D image
 * 
 * @return The summation of each superimposed product summation of each channel.
 */
float sup_product_summation_3d(Kernel* channel_kerns, float** img3d, int img3d_ncols, int sup_start_row, int sup_start_col, int nchannels) {
    float result = 0;

    for (int ch = 0; ch < nchannels; ++ch) {
        result += channel_kerns[ch].sup_prod_sum(channel_kerns[ch].matrix, img3d[ch], channel_kerns[ch].nrows, channel_kerns[ch].ncols, 
                                                 img3d_ncols, sup_start_row, sup_start_col);
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


/* CONVOLUTIONAL BUILDER FUNCTIONS */

/*
 * Allocates memory for the convolutional layer attribute storing a list of kernels, if the number of channels is equal to one.
 * Otherwise, allocates memory for hte convolutional layer attribute storing a list of 3D kernels.
 * 
 * @param kernels_ref | A reference to the convolutional layer attribute that stores a list of kernels
 * @param kernels3d_ref | A reference to the convolutional layer attribute that stores a list of 3D kernels
 * @param num_kernels | The specified number of kernels to be included in the convolutional layer being initialized
 * @param num_channels | The specified number of channels to be included in the convolutional layer being initialized
 */
void alloc_for_ref_kern_lists(Kernel** kernels_ref, Kernel3D** kernels3d_ref, int num_kernels, int num_channels) {
    if (num_channels == 1) { 
        *kernels_ref = (Kernel*) malloc(sizeof(Kernel) * num_kernels);
        *kernels3d_ref = NULL;
    } else {
        *kernels_ref = NULL;
        *kernels3d_ref = (Kernel3D*) malloc(sizeof(Kernel3D) * num_kernels);
    }
}

/*
 * Initializes the given kernel matrix with the specified kernel elements, if they are given. If the specified kernel elements aren't
 * given, then initializes the given kernel matrix with elements randomly generated between -2 and 2.
 * 
 * @param kmatrix | The matrix associated with the given kernel
 * @param kern_nrows | The number of rows for each kernel in the convolutional layer being initialized
 * @param kern_ncols | The number of columns for each kernel in the convolutional layer being initialized
 * @param kern_elems | The specified elements, if any, to be stored in each matrix in each kernel. If aren't any, this param equals NULL
 */
void init_kern_mat(float* kmatrix, int kern_nrows, int kern_ncols, float* kern_elems) {
    for (int i = 0; i < kern_nrows; ++i) {
        for (int j = 0; j < kern_ncols; ++j) {
            if (kern_elems) {
                set_mat_val(kmatrix, kern_ncols, i, j, mat_val(kern_elems, kern_ncols, i, j));
            } else {
                set_mat_val(kmatrix, kern_ncols, i, j, random_num(-2, 2, 3));
            }
        }
    }
}

/*
 * Initializes the kernel associated with a specific channel to adopt the elements of a given matrix with given dimensions.
 *
 * @param channel_kernels | A list of kernels, each associated with a specific channel
 * @param kmatrix | The matrix containing the elements the initializing kernel will adopt
 * @param kern_nrows | The number of rows for the kernel being initialized
 * @param kern_ncols | The number of columns for the kernel being initialized
 */
void init_channel_kernel(Kernel* channel_kernels, int channel, float* kmatrix, int kern_nrows, int kern_ncols) {
    channel_kernels[channel].matrix = kmatrix;
    channel_kernels[channel].nrows = kern_nrows;
    channel_kernels[channel].ncols = kern_ncols;
    channel_kernels[channel].sup_prod_sum = sup_product_summation;
}

/*
 * If there is only one channel, adds the only kernel in the channel kernels list to the referenced convolutional layer attribute
 * storing a list of kernels. Otherwise, adds a newly initialized 3D kernel, which consists of every kernel in the channel kernels
 * list to the referenced convolutional layer attribute storing a list of 3D kernels.
 * 
 * @param kernels_ref | A reference to the convolutional layer attribute that stores a list of kernels
 * @param kernels3d_ref | A reference to the convolutional layer attribute that stores a list of 3D kernels
 * @param k | The index at which the new kernel should be added
 * @param channel_kernels | A list of kernels related to each channel, where this list correlates to the kth, the newest, kernel addition
 * @param num_channels | The specified number of channels of within the new kernel being added
 */
void add_new_kernel_to_kern_lists(Kernel** kernels_ref, Kernel3D** kernels3d_ref, int k, Kernel* channel_kernels, int num_channels) {
    if (num_channels == 1) {
        (*kernels_ref)[k] = channel_kernels[0];
    } else {
        (*kernels3d_ref)[k].channel_kerns = channel_kernels;
        (*kernels3d_ref)[k].num_channels = num_channels;
        (*kernels3d_ref)[k].sup_prod_sum_3d = sup_product_summation_3d;
    }
}

/*
 * Initializes the fields, which are not function pointers, for a convolutional layer object based on a specified number of kernels, 
 * a specified number of channels, specified kernel dimensions, and, optionally, specified elements to be stored in each kernel's matrix, 
 * if there is only one channel, or matrices, if there is more than one channel. If the elements to be stored are not specified, then the 
 * elements to be stored in each kernel's matrix, if there is only one channel, or matrices, if there is more than one channel, will be 
 * randomly generated between -2 and 2.
 * 
 * The fields to be initialized are:
 *  -> A list of kernels, which will contain a specified number of kernels – square matrices of specified row and column dimensions
 *      • If the specified number of channels is greater than one, then this field will be left NULL
 * 
 *  -> A list of 3D kernels, which will contain a specified number of 3D kernels – 3D square matrices with width equal to num of channels
 *      • If the specified number of channels is equal to one, then this field will be left NULL
 * 
 *  -> An integer representing the number of kernels
 *  -> An integer representing the  number of channels
 * 
 * 
 * @param kernels_ref | A reference to the convolutional layer attribute that stores a list of kernels
 * @param kernels3d_ref | A reference to the convolutional layer attribute that stores a list of 3D kernels
 * @param num_kernels_ref | A reference to the convolutional layer attribute that stores an integer representing the number of kernels
 * @param num_channels_ref | A reference to the convolutional layer attribute that stores an integer representing the number of channels
 * @param num_kerns | The specified number of kernels to be included in the convolutional layer being initialized
 * @param num_channels | The specified number of channels to be included in the convolutional layer being initialized
 * @param kern_nrows | The number of rows for each kernel in the convolutional layer being initialized
 * @param kern_ncols | The number of columns for each kernel in the convolutional layer being initialized
 * @param kern_elems | The specified elements, if any, to be stored in each matrix in each kernel. If aren't any, this param equals NULL
 */
void convl_builder(Kernel** kernels_ref, Kernel3D** kernels3d_ref, int* num_kernels_ref, int* num_channels_ref, 
                   int num_kerns, int num_channels, int kern_nrows, int kern_ncols, float* kern_elems) {

    *num_kernels_ref = num_kerns;
    *num_channels_ref = num_channels;

    alloc_for_ref_kern_lists(kernels_ref, kernels3d_ref, num_kerns, num_channels);

    for (int k = 0; k < num_kerns; ++k) {
        Kernel* channel_kernels = (Kernel*) malloc(sizeof(Kernel) * num_channels);

        for (int ch = 0; ch < num_channels; ++ch) {
            float* kmatrix = (float*) malloc(sizeof(float) * kern_nrows * kern_ncols);
            init_kern_mat(kmatrix, kern_nrows, kern_ncols, kern_elems);
            init_channel_kernel(channel_kernels, ch, kmatrix, kern_nrows, kern_ncols);
        }

        add_new_kernel_to_kern_lists(kernels_ref, kernels3d_ref, k, channel_kernels, num_channels);
    }
}

/* CONVOLUTION EXECUTION FUNCTIONS */

/*
 * Applies the row-wise padding and column-wise padding specified to the given image. Assumes zero padding is used.
 *
 * @param img | The image to which padding is being added
 * @param img_nrows | The number of rows in the image being modified
 * @param img_ncols | The number of columns in the image being modified
 * @param num_channels | The number of channels in the input image
 * @param padding | Contains the amount of padding applied row-wise and column-wise
 * 
 * @return The image with the specified padding applied
 */
float** img_with_padding(float** img, int img_nrows, int img_ncols, int num_channels, RowColTuple padding) {
    float** pad_img = (float**) malloc(sizeof(float*) * (img_nrows + 2 * padding.rows));

    for (int ch = 0; ch < num_channels; ++ch) {
        float* pad_img_ch = (float*) malloc(sizeof(float) * (img_nrows + 2 * padding.rows));

        for (int r = 0; r < img_nrows + 2 * padding.rows; ++r) {
            for (int c = 0; c < img_ncols + 2 * padding.cols; ++c) {
                if (r < padding.rows || r >= img_nrows + padding.rows) {
                    set_mat_val(pad_img_ch, img_ncols + 2 * padding.cols, r, c, 0);
                } else if (c < padding.cols || c >= img_ncols + padding.cols) {
                    set_mat_val(pad_img_ch, img_ncols + 2 * padding.cols, r, c, 0);
                } else {
                    set_mat_val(pad_img_ch, img_ncols + 2 * padding.cols, r, c, mat_val(img[ch], img_ncols, r - padding.rows, c - padding.cols));
                }
            }
        }

        pad_img[ch] = pad_img_ch;
    }

    return pad_img;
}

int calc_output_dimen_size(int img_dimen_size, int kern_dimen_size, int padding_for_dimen, int stride_for_dimen) {
    return floorf(((img_dimen_size + 2 * padding_for_dimen - kern_dimen_size) / stride_for_dimen) + 1);
}

float* convolution(void* kernels_nd, int k, int kern_nrows, int kern_ncols, int num_channels,
                   float** aug_img, int aug_img_ncols, int output_nrows, int output_ncols, RowColTuple stride) {

    
    Kernel* kernels = NULL;
    Kernel3D* kernels_3d = NULL;

    if (num_channels == 1) {
        kernels = (Kernel*) kernels_nd;
    } else {
        kernels_3d = (Kernel3D*) kernels_nd;
    }
    
    float* output_img = (float*) malloc(sizeof(float) * output_nrows * output_ncols);

    for (int sup_r = 0, out_r = 0; out_r < output_nrows; sup_r += stride.rows, ++out_r) {
        for (int sup_c = 0, out_c = 0; out_c < output_ncols; sup_c += stride.cols, ++out_c) {
            if (kernels) {
                set_mat_val(output_img, output_ncols, out_r, out_c, 
                            kernels[k].sup_prod_sum(kernels[k].matrix, *aug_img, 
                                                    kernels[k].nrows, kernels[k].ncols, aug_img_ncols, sup_r, sup_c));
            } else {
                set_mat_val(output_img, output_ncols, out_r, out_c, 
                            kernels_3d[k].sup_prod_sum_3d(kernels_3d[k].channel_kerns, aug_img, aug_img_ncols, sup_r, sup_c, num_channels));
            }
        }
    }

    return output_img;
}

/*
 * Conducts an entire convolution operation over the image being processed, where each kernel produces an output image resulting from 
 * the convolution operation.
 * 
 * @param kernels | A list of kernels
 * @param kernels3d | A list of 3D kernels
 * @param num_kernels | The number of kernels convolving over the given image
 * @param num_channels | The number of channels in each kernel and the given image
 * @param kern_nrows | The number of rows for each kernel convolving over the given image
 * @param kern_ncols | The number of columns for each kernel convolving over the given image
 * @param img | The image, which is represented as a tensor of floating-point numbers, that's being convolved over
 * @param img_nrows | The number of rows in the image
 * @param img_ncols | The number of columns in the image
 * @param padding | A specifier indicating whether there should or should not be any padding
 * @param stride | The amount by which the kernel will slide over the image during each convolution iteration
 * 
 * @return A tensor of output images resulting from their respective kernels' convolution operations, each being represented as a matrix
 */
float** convl_exec(Kernel* kernels, Kernel3D* kernels3d, int num_kernels, int num_channels, int kern_nrows, int kern_ncols, 
                  float** img, int img_nrows, int img_ncols, RowColTuple padding, RowColTuple stride) {

    float** aug_img = img_with_padding(img, img_nrows, img_ncols, num_channels, padding);
    int aug_img_nrows = img_nrows + 2 * padding.rows;
    int aug_img_ncols = img_ncols + 2 * padding.cols;

    int output_nrows = calc_output_dimen_size(img_nrows, kern_nrows, padding.rows, stride.cols);
    int output_ncols = calc_output_dimen_size(img_ncols, kern_ncols, padding.cols, stride.cols);

    float** output_imgs = (float**) malloc(sizeof(float*) * num_kernels);

    for (int k = 0; k < num_kernels; ++k) {
        // if (kernels)
        //  float* output_img = convolution(kernels, k, kern_nrows, kern_ncols, num_channels,
        //                                  aug_img, aug_img_ncols, output_nrows, output_ncols, stride);
        float* output_img = (float*) malloc(sizeof(float) * output_nrows * output_ncols);

        for (int sup_r = 0, out_r = 0; sup_r + kern_nrows <= img_nrows; sup_r += stride.rows, ++out_r) {
            for (int sup_c = 0, out_c = 0; sup_c + kern_ncols <= img_ncols; sup_c += stride.cols, ++out_c) {
                if (kernels) {
                    set_mat_val(output_img, output_ncols, out_r, out_c, 
                                kernels[k].sup_prod_sum(kernels[k].matrix, aug_img[0], 
                                                        kernels[k].nrows, kernels[k].ncols, img_ncols, sup_r, sup_c));
                } else {
                    set_mat_val(output_img, output_ncols, out_r, out_c, 
                                kernels3d[k].sup_prod_sum_3d(kernels3d[k].channel_kerns, aug_img, img_ncols, sup_r, sup_c, num_channels));
                }
            }
        }

        output_imgs[k] = output_img;
    }

    return output_imgs;
}

typedef struct convo_layer {
    Kernel* kernels;
    Kernel3D* kernels3d;

    int num_kernels;
    int num_channels;

    void (*build) (Kernel**, Kernel3D**, int*, int*, int, int, int, int, float*);
    float** (*exec) (Kernel*, Kernel3D*, int, int, int, int, float**, int, int, RowColTuple, RowColTuple);
} ConvoLayer;

#endif