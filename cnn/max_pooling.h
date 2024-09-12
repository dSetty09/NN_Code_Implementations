/*** CONSISTS OF DEFINITION NECESSARY FOR MAX POOLING OPERATION OF A MAX POOLING LAYER TO BE CONDUCTED ***/

#ifndef MAX_POOLING_H
#define MAX_POOLING_H

/*
 * Max function for calculating the maximum cell value of all cell values of a superimposed area for max pooling. 
 *
 * @param kern_mat | The kernel matrix being passed over the superimposed part of the given image (all cell values take up the value 1)
 * @param img | The image max pooling being conducted over
 * @param kern_nrows | The number of rows in the kernel matrix
 * @param kern_ncols | The number of columns in the kernel matrix
 * @param img_ncols | The number of columns in the given image
 * @param sup_start_row | The starting row index for the superimposed part of the image
 * @param sup_start_col | The starting column index for the superimposed part of the image
 * 
 * @return The maximum value across all values in each cell of the part of the image superimposed by the kernel
 */
float sup_max(float* kern_mat, float* img, int kern_nrows, int kern_ncols, int img_ncols, int sup_start_row, int sup_start_col) {
    float max = mat_val(img, img_ncols, sup_start_row, sup_start_col);

    for (int kern_i = 0, sup_i = sup_start_row; kern_i < kern_nrows; ++kern_i, ++sup_i) {
        for (int kern_j = 0, sup_j = sup_start_col; kern_j < kern_ncols; ++kern_j, ++sup_j) {
            float curr_sup_val = mat_val(img, img_ncols, sup_start_row, sup_start_col);
            
            if (curr_sup_val > max) {
                max = curr_sup_val;
            }
        }
    }

    return max;
}

#endif