#include <stdio.h>
#include <assert.h>
#include "convo_layer.h"

int main() {
    /* KERNEL OPS TESTS */
    // float kern_mat[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    

    // int kern_dimen = 3;

    // Kernel kernel = {kern_mat, kern_dimen, sup_product_summation};

    // float sup_mat1[] = {2, -2, 1, 0, -1, 0, 2, 1, 2};

    // int sup_mat_nrows = 3;
    // int sup_mat_ncols = 3;

    // int sup_mat_centrow = 1;
    // int sup_mat_centcol = 1;

    // float convoresult2d = kernel.sup_prod_sum(kernel.matrix, sup_mat1, kern_dimen, sup_mat_ncols, sup_mat_centrow, sup_mat_centcol);

    // printf("2D Kernel Convolution Expected Result:%f\n", -1.0);
    // printf("2D Kernel Convolution Actual Result:%f\n", convoresult2d);
    // assert(convoresult2d == -1);
    // printf("2D Kernel Convolution Test Passed.\n\n");

    /* 3D KERNEL OPS TESTS */

    // float sup_mat2[] = {3, -2, 5, 0, -7, 0, 8, 1, -3};
    // float sup_mat3[] = {4, -1, 4, 0, -6, 0, 7, 0, -2};

    // Kernel kernels[] = {kernel, kernel, kernel};
    // int num_channels = 3;

    // Kernel3D kernel3d = {kernels, num_channels, sup_product_summation_3d};

    // float* sup_mat3d[] = {sup_mat1, sup_mat2, sup_mat3};

    // float convoresult3d = kernel3d.sup_prod_sum_3d(kernel3d.channel_kerns, sup_mat3d, sup_mat_ncols, sup_mat_centrow, sup_mat_centcol, num_channels);

    // printf("3D Kernel Convolution Expected Result:%f\n", -19.0);
    // printf("3D Kernel Convolution Actual Result:%f\n", convoresult3d);
    // assert(convoresult3d == -19);
    // printf("3D Kernel Convolution Test Passed.\n");

    /* CONVOLUTIONAL LAYER TESTS */

    float** img = (float**) malloc(sizeof(float));
    float subimg[] = {-8, 5, 1, 7, 2, 0, 1, 3, 5, 6, 2, 2, 1, 5, 3, 0, 4, 3, 7, 2, 1, 1, -7, -6, 5};
    *img = subimg;

    float kern_elems[] = {1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1};

    ConvoLayer convl; convl.build = convl_builder; convl.exec = convl_exec;
    convl.build(&convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 4, 4, kern_elems);

    // no padding, stride of 1
    RowColTuple padding = {0, 0};
    RowColTuple stride = {1, 1};

    float correct_result_img[] = {-18, 7, 0, -8};

    float** convl_result_imgs = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 4, 4, img, 5, 5, padding, stride);
    float* convl_result_img = *convl_result_imgs;

    assert(mats_equal(correct_result_img, convl_result_img, 2, 2));
    printf("Convolution works properly on square image.\n");

    free(img);
}