/* Notes:
 * This file contains common type definitions used across all C files in all folders
 */

#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <assert.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>

/* VARIABLE DEFINITIONS */

#define TRUE 1
#define FALSE 0

static const unsigned char OUTPUT = 'O';
static const unsigned char WEIGHT = 'W';
static const unsigned char BIAS = 'B';

static const int NO_DERIV = -1;

static const float NEAR_ZERO = 1E-15;

static const float BP_LOG_MIN_INPUT = 0;
static const float BP_LOG_MAX_INPUT = FLT_MAX;
static const float BP_LOG_MAX_OUTPUT = 88.722839;

static const float BP_SQUARE_MAX_INPUT = 18446742974197923840.000000;
static const float BP_SQUARE_MIN_INPUT = -18446742974197923840.000000;
static const float BP_SQUARE_MAX_DERIV_INPUT = 170141173319264429905852091742258462720.000000;
static const float BP_SQUARE_MIN_DERIV_INPUT = -170141173319264429905852091742258462720.000000;

typedef float (*one_arg_activation_function) (float, int);
typedef float (*cost_function) (float[], float[], int, int);


/* STRUCT DEFINITIONS */

/* A tuple consisting of only two integer elements */
typedef struct rctuple {
    int rows; // The first element corresponding to rows
    int cols; // The second element corresponding to columns
} RowColTuple;

typedef struct rvf {
    void* vect;

    int num_rows;
    int num_cols;
    int num_layers;
} ReadVectFmt;


/* FUNCTION DEFINITIONS */

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

float floor_to_place(float val, float place) {
    float multiple = powf(10, place);
    return floorf(val * multiple) / multiple;
}

int are_similar(float val1, float val2) {
    if (abs(val1 - val2) < 0.00001) {
        return 1;
    }

    return 0;
}

float arr_max(int* arr, int num) {
    float max = arr[0];

    for (int i = 0; i < num; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }

    return max;
}

float bp_safe_exp(float x, int deriv) {
    if (x <= FLT_MAX_10_EXP && x >= FLT_MIN_10_EXP) return (deriv) ? expf(x) + NEAR_ZERO : expf(x) + NEAR_ZERO;

    if (x < FLT_MIN_10_EXP) return NEAR_ZERO;

    return (deriv) ? FLT_MAX : FLT_MAX;
}

float bp_safe_log(float x, int deriv) {
    if (x <= BP_LOG_MAX_INPUT && x >= BP_LOG_MIN_INPUT) return (deriv) ? (1 / x + NEAR_ZERO) : logf(x + NEAR_ZERO);

    if (x < BP_LOG_MIN_INPUT) return NAN;

    return (deriv) ? NEAR_ZERO : BP_LOG_MAX_OUTPUT;
}

float bp_safe_square(float x, int deriv) {
    if (x <= BP_SQUARE_MAX_INPUT && x >= BP_SQUARE_MIN_INPUT) return (deriv) ? (2 * x) : powf(x, 2);

    if (x < BP_SQUARE_MIN_INPUT && deriv) return (x < BP_SQUARE_MIN_DERIV_INPUT) ? (2 * BP_SQUARE_MIN_DERIV_INPUT) : (2 * x);

    if (deriv) return (x > BP_SQUARE_MAX_DERIV_INPUT) ? (2 * BP_SQUARE_MAX_DERIV_INPUT) : (2 * x);

    return FLT_MAX;
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

/*
 * Checks if two matrices are equal (i.e. they contain the same elements for each cell). Assumes that the two matrices being
 * compared have the same dimensions.
 *
 * @param mat1 | The first matrix being compared
 * @param mat2 | The second matrix being compared
 * @param nrows | The number of rows in each matrix
 * @param ncols | The number of columns in each matrix
 * 
 * @return 1 if both the matrices are deemed equal, and 0 otherwise
 */
int mats_equal(float* mat1, float* mat2, int nrows, int ncols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            float first_val = round_to_place(mat_val(mat1, ncols, i, j), 5);
            float second_val = round_to_place(mat_val(mat2, ncols, i, j), 5);

            if (first_val != second_val) {
                return 0;
            }
        }
    }

    return 1;
}

/*
 * Flattens a given set of feature maps into a one dimensional vector. If the given set of feature maps
 * contains only one feature map (i.e. there is only one channel) then no flattening occurs, since an image
 * or feature map is already represented as a one dimensional vector in terms of this CNN implementation
 * 
 * @param feature_map_set | 
 */
float* flatten_img(void* img, int num_rows, int num_cols, int num_channels) {
    if (num_channels == 1) {
        return (float*) img;
    }

    float* flattened_set = (float*) malloc(sizeof(float) * num_channels * num_rows * num_cols);
    int entries_per_img_channel = num_rows * num_cols;

    for (int ch = 0; ch < num_channels; ++ch) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                flattened_set[entries_per_img_channel * ch + num_cols * r + c] = 
                    mat_val(((float**) img)[ch], num_cols, r, c);
            }
        }
    }

    return flattened_set;
}

/*
 * Finds the minimum element in a vector.
 *
 * @param vect | The vector being searched
 * @param num_rows | The number of rows in the vector (i.e. the 1st dimension of the vector)
 * @param num_cols | The number of columns in the vector (i.e. the 2nd dimension of the vector)
 * @param num_layers | The number of layers in the vector (i.e. the 3rd dimension of the vector)
 * 
 * @return The minimum element in the given vector
 */
float min_vect_elem(void* vect, int num_rows, int num_cols, int num_layers) {
    float** vect3d = NULL;
    float* vect2d = NULL;

    if (num_layers > 1) {
        vect3d = (float**) vect;
    } else {
        vect2d = (float*) vect;
    }

    float min = (vect3d) ? mat_val(vect3d[0], num_rows, 0, 0) : mat_val(vect2d, num_cols, 0, 0);

    if (vect2d) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                float curr_val = mat_val(vect2d, num_cols, r, c);
                if (curr_val < min) min = curr_val;
            }
        }

        return min;
    }

    for (int l = 0; l < num_layers; ++l) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                float curr_val = mat_val(vect3d[l], num_cols, r, c);
                if (curr_val < min) min = curr_val;
            }
        }
    }

    return min;
}

/*
 * Prints a vector in a readable format.
 *
 * @param vect | The vector to be printed
 * @param num_rows | The number of rows in the vector (i.e. the 1st dimension of the vector)
 * @param num_cols | The number of columns in the vector (i.e. the 2nd dimension of the vector)
 * @param num_layers | The number of layers in the vector (i.e. the 3rd dimension of the vector)
 * @param max_num_places | The max number of places that an entry in the vector has
 * @param max_nth_places | The max number of nth places that an entry in the vector has
 * @param vect_name | The name of the vector being printed
 * @param vect_name_len | The length, in characters, of the name of the vector being printed
 * @param output_file | The output file the vector is being printed out to
 */
void print_vector(void* vect, int num_rows, int num_cols, int num_layers,
                  const char* vect_name, int vect_name_len, FILE* output_file) {

    int vect_offset = vect_name_len + 3;
    int vect_centr_index = num_rows / 2;

    int elem_offset; elem_offset = (min_vect_elem(vect, num_rows, num_cols, num_layers) < 0) ? 1 : 0;

    for (int r = 0; r < num_rows; ++r) {
        if (r == vect_centr_index) fprintf(output_file, "%s = ", vect_name);
        else for (int i = 0; i < vect_offset; ++i) fprintf(output_file, " ");

        for (int l = 0; l < num_layers; ++l) {
            fprintf(output_file, "[");

            for (int c = 0; c < num_cols; ++c) {
                float entry_to_print; 

                if (num_layers > 1) {
                    entry_to_print = mat_val(((float**) vect)[l], num_cols, r, c);
                } else {
                    entry_to_print = mat_val((float*) vect, num_cols, r, c);
                }

                for (int i = 0; i < elem_offset; ++i) fprintf(output_file, " ");
                fprintf(output_file, "%f", entry_to_print);
                
                if (c < num_cols - 1) fprintf(output_file, ", ");
            }

            fprintf(output_file, "]");
            
            if (l == num_layers - 1) {
                fprintf(output_file, "\n");
            } else {
                if (r == vect_centr_index) fprintf(output_file, " , ");
                else fprintf(output_file, "   ");
            }
        }
    }
}

/* 
 * Displays results of a certain test for a certain function in a readable format
 *
 * @param testing_function | Name of function being tested
 * @param conducting_test | Name of test being conducted
 * @param expected | The expected value of the test
 * @param actual | The actual value of the test
 * @param exp | The expression included in the assertion
 * @param testing_value | A flag indicating whether we are testing a whole value or not. If not, we are testing array values.
 * @param output_file | The file writing test results to
 */
void disp_test_results(const char* testing_function, const char* conducting_test, void* expected, void* actual, int testing_value, FILE* output_file) {
    fprintf(output_file, "-----------------------------------------------------\n");
    fprintf(output_file, "%s | %s\n", testing_function, conducting_test);

    if (testing_value) {
        fprintf(output_file, "Expected Result: %f\n", *((float*) expected));
        fprintf(output_file, "Actual Result: %f\n", *((float*) actual));
        assert(floor_to_place(*((float*) expected), 5) == floor_to_place(*((float*) actual), 5));
    } else {
        ReadVectFmt* expected_rvf = (ReadVectFmt*) expected;
        print_vector(expected_rvf->vect, expected_rvf->num_rows, expected_rvf->num_cols, expected_rvf->num_layers,
                     "Expected", 8, output_file);

        fprintf(output_file, "\n");

        ReadVectFmt* actual_rvf = (ReadVectFmt*) actual;
        print_vector(actual_rvf->vect, actual_rvf->num_rows, actual_rvf->num_cols, actual_rvf->num_layers,
                     "Actual", 6, output_file);

        assert(mats_equal((float*) expected_rvf->vect, (float*) actual_rvf->vect, 
                          expected_rvf->num_rows, expected_rvf->num_cols));
    }

    fprintf(output_file, "Test passed.\n");
    fprintf(output_file, "-----------------------------------------------------\n");
}


#endif