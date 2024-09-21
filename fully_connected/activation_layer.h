/*** 
 *** CONSISTS OF FUNCTIONS AND STRUCTURES NECESSARY FOR IMPLENTING LAYER THAT PASSES INPUT THROUGH ACTIVATION 
 *** LAYER  
 ***/

#include "../common_definitions.h"
#include "../activation_functions/nonlinear_functions.h"

#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H


/** ACTIVATION LAYER DEFINITIONS **/

float* gen_output_img(void* img, int nrows, int ncols, int nchannels, one_arg_activation_function act_func) {
    if (nchannels > 1) return NULL;

    float* output_img = (float*) malloc(sizeof(float) * nrows * ncols); 

    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            set_mat_val(output_img, ncols, r, c, act_func(mat_val((float*) img, ncols, r, c), FALSE));
        }
    }

    return output_img;
}

float** gen_3d_output_img(void* img, int nrows, int ncols, int nchannels, one_arg_activation_function act_func) {
    if (nchannels == 1) return NULL;

    float** output_img = (float**) malloc(sizeof(float*) * nchannels);

    for (int ch = 0; ch < nchannels; ++ch) {
        output_img[ch] = gen_output_img((void*) ((float**) img)[ch], nrows, ncols, nchannels, act_func);
    }

    return output_img;
}

/* The activation layer in a CNN */
typedef struct activation_layer {
    void* img; // the input image being filtered through this layer's activation function

    int num_rows; // the number of rows in the input image
    int num_cols; // the number of columns in the input image
    int num_channels; // the number of channels in the input image

    one_arg_activation_function activation_function; // the activation function for this layer

    float* (*gen_output_img) (void*, int, int, int, one_arg_activation_function);
    float** (*gen_3d_output_img) (void*, int, int, int, one_arg_activation_function);
} ActLayer;

#endif