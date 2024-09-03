/* Notes:
 * This file contains common type definitions used across all C files in all folders
 */

#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRUE 1
#define FALSE 0

typedef float (*one_arg_activation_function) (float, int);
typedef float (*cost_function) (float[], float[], int, int);

/*
 * Rounds a specific value to a certain number of decimal places.
 *
 * @param val | The floating point number being rounded
 * @param place | The decimal place to round to
 * 
 * @return The result of rounding a specific value to a certain number of decimal places
 */
float round_to_place(float val, float place) {
    float multiple = powf(10, place);
    return round(val * multiple) / multiple;
}

/*
 * Generates a random number between a minimum and maximum value, inclusive. 
 *
 * @param min | The minimum value
 * @param max | The maximum value
 * @param precision | The decimal precision of the minimum value and maximum value
 * 
 * @return A random number between a minimum value and a maximum value.
 */
float random_num(float min, float max, float precision) {
    srand(time(NULL)); // set random number generator seed based on current time

    int multiple = powf(10, precision);

    int temp_min = (int) ceilf((min * multiple));
    int temp_max = (int) floorf(max * multiple) + 1;

    int shift_val = -temp_min;

    if (min < 0) {
        temp_min += shift_val;
        temp_max += shift_val;
    }

    int temp_ret = (rand() + temp_min) % temp_max;
    
    if (min < 0) {
        temp_ret -= shift_val;
    }

    float ret = ((float) temp_ret) / ((float) multiple);
    return ret;
}

/*
 * Retrieves the value at a specific row and column index in a matrix.
 *
 * @param mat | The matrix from which we are retrieving a value
 * @param ncols | The number of columns in the aforementioned matrix
 * @param i | The index of the row from which we are retrieving from
 * @param j | The index of the column from which we are retrieving from
 * 
 * @return The value at the ith row index and jth column index in a specific matrix
 */
float mat_val(float* mat, int ncols, int i, int j) {
    return mat[ncols * i + j];
}

/*
 * Sets the value at a specific row and column index in a matrix to some other value.
 *
 * @param mat | The matrix in which we are storing a new value
 * @param ncols | The number of columns in the aforementioned matrix
 * @param i | The index of the row from which we are retrieving from
 * @param j | The index of the column from which we are retrieving from
 * @param new_val | The new value being stored
 * 
 * @return Nothing
 */
void set_mat_val(float* mat, int ncols, int i, int j, float new_val) {
    mat[ncols * i + j] = new_val;
}

#endif