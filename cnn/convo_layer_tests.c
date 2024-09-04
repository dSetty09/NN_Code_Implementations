#include <stdio.h>
#include <assert.h>
#include "convo_layer.h"

int main() {
    /* KERNEL OPS TESTS */
    float kern_mat[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    

    int kern_dimen = 3;

    Kernel kernel = {kern_mat, kern_dimen, sup_product_summation};

    float sup_mat1[] = {2, -2, 1, 0, -1, 0, 2, 1, 2};

    int sup_mat_nrows = 3;
    int sup_mat_ncols = 3;

    int sup_mat_centrow = 1;
    int sup_mat_centcol = 1;

    float convoresult2d = kernel.sup_prod_sum(kernel.matrix, sup_mat1, kern_dimen, sup_mat_ncols, sup_mat_centrow, sup_mat_centcol);

    printf("2D Kernel Convolution Expected Result:%f\n", -1.0);
    printf("2D Kernel Convolution Actual Result:%f\n", convoresult2d);
    assert(convoresult2d == -1);
    printf("2D Kernel Convolution Test Passed.\n\n");

    /* 3D KERNEL OPS TESTS */

    float sup_mat2[] = {3, -2, 5, 0, -7, 0, 8, 1, -3};
    float sup_mat3[] = {4, -1, 4, 0, -6, 0, 7, 0, -2};

    Kernel kernels[] = {kernel, kernel, kernel};
    int num_channels = 3;

    Kernel3D kernel3d = {kernels, num_channels, sup_product_summation_3d};

    float* sup_mat3d[] = {sup_mat1, sup_mat2, sup_mat3};

    float convoresult3d = kernel3d.sup_prod_sum_3d(kernel3d.channel_kerns, sup_mat3d, sup_mat_ncols, sup_mat_centrow, sup_mat_centcol, num_channels);

    printf("3D Kernel Convolution Expected Result:%f\n", -19.0);
    printf("3D Kernel Convolution Actual Result:%f\n", convoresult3d);
    assert(convoresult3d == -19);
    printf("3D Kernel Convolution Test Passed.\n");

}