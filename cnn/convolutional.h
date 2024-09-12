/*** CONSISTS OF DEFINITION NECESSARY FOR CONVOLUTION OPERATION OF A CONVOLUTIONAL LAYER TO BE CONDUCTED ***/

#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H

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
            result += mat_val(kern_mat, kern_ncols, kern_i, kern_j) * mat_val(img, img_ncols, sup_i, sup_j);
        }
    }

    return result;
}

#endif