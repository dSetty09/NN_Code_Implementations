#include <stdio.h>
#include <assert.h>
#include "convo_layer.h"

int main() {
    /* KERNEL OPS TESTS */
    float kern_mat[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    

    int kernnrows = 3;
    int kernncols = 3;

    Kernel kernel = {kern_mat, kernnrows, kernncols, convolution};

    float sup_mat1[] = {2, -2, 1, 0, -1, 0, 2, 1, 2};
    float sup_mat2[] = {3, -2, 5, 0, -7, 0, 8, 1, -3};
    float sup_mat3[] = {4, -1, 4, 0, -6, 0, 7, 0, -2};

    float convoresult2d = kernel.convo_func(kernel.matrix, sup_mat1, kernel.nrows, kernel.ncols);

    printf("2D Kernel Convolution Expected Result:%f\n", -1.0);
    printf("2D Kernel Convolution Actual Result:%f\n", convoresult2d);
    assert(convoresult2d == -1);
    printf("2D Kernel Convolution Test Passed.\n\n");

    /* 3D KERNEL OPS TESTS */

    Kernel kernels[] = {kernel, kernel, kernel};
    int nkerns = 3;

    Kernel3D kernel3d = {kernels, nkerns, convolution3d};

    float* sup_mat3d[] = {sup_mat1, sup_mat2, sup_mat3};

    float convoresult3d = kernel3d.convo_3d(kernel3d.channel_kerns, kernel3d.num_channels, sup_mat3d);

    printf("3D Kernel Convolution Expected Result:%f\n", -19.0);
    printf("3D Kernel Convolution Actual Result:%f\n", convoresult3d);
    assert(convoresult3d == -19);
    printf("3D Kernel Convolution Test Passed.\n");

}