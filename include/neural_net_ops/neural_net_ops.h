/* Notes:
 * This file contains common type definitions used for all neural network implementations 
 */

#ifndef NEURAL_NET_OPS_H
#define NEURAL_NET_OPS_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


/* VARIABLE DEFINITIONS */

static const unsigned int TRUE = 1;
static const unsigned int FALSE = 0;

static const unsigned int PAST_MAX_EPOCHS = 0;
static const unsigned int PAST_MIN_DIFF = 1;
static const unsigned int PAST_FAIR_ERROR = 2;

static const unsigned char OUTPUT = 'O';
static const unsigned char WEIGHT = 'W';
static const unsigned char BIAS = 'B';

static const int NO_DERIV = -1;

static const float NEAR_ZERO = 1e-6;

static const float MAX_FLT_EXP = 88.722839;
static const float MIN_FLT_EXP = -15.942385;

static const float MIN_FLT_LOG_ARG = 1e-45;
static const float MAX_FLT_LOG_ARG = 3e38;

static const float MIN_FLT_SQR_ARG = -18446743523953729536.000000;
static const float MAX_FLT_SQR_ARG = 18446743523953729536.000000;

/* TYPE DEFINITIONS */
typedef float (*simple_act_func) (float, int);
typedef float (*softmax_act_func) (float*, int, unsigned int, int);

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
float round_to_place(float val, float place); 

/*
 * Compares two float values to check if they are equal, taking float rounding inaccuracies 
 * into account.
 * 
 * @param val1 | The first float value being compared
 * @param val2 | The second float value being compared
 * 
 * @return 1 if they are deemed equal and 0 otherwise
 */
int are_equal(float val1, float val2);

/*
 * Retrieves the largest value in a float array.
 *
 * @param arr | The float array
 * @param num | The number of elements in said float array
 * 
 * @return the largest value in the float array
 */
float arr_max(float* arr, int num); 

/*
 * Creates a deep copy from a specific array to
 * a certain array.
 * 
 * @param from | The array deep copying from
 * @param to | The array deep copying to
 * @param n | The size of the array being duplicated
 */
void arr_dup(float* from, float** to, int n);

/*
 * Deletes the memory allocated for a specified array.
 *
 * @param arr | The array whose allocated memory is being freed.
 * 
 * @warning Only use for arrays that are dynamically allocated.
 */
void arr_delete(float* arr);

/*
 * Prints the content of a given array.
 *
 * @param arr | The array whose content is being printed.
 * @param n | The size of the given array.
 */
void arr_print(float* arr, int n);

/*
 * Conducts an exponent operation that is "safe" for float operations. 
 * In other words, this function will never output a value that is greater
 * than the maximum representable float value and a value that is less than
 * the smallest representable float value greater than zero.
 * 
 * @param x | The x value
 * 
 * @return e^x such that the value outputted is safe for float arithmetic operations. 
 */
float flt_safe_exp(float x);

/*
 * Similar to above function, except that it ensures that log operation is safe for 
 * float arithmetic operations. 
 * 
 * @param x | The x value
 * 
 * @return log(x) such that the value outputted is safe for float arithmetic operations.
 */
float flt_safe_log(float x);

/*
 * Similar to above function, except that it ensures that square operation is "safe" for 
 * float arithmetic operations.
 * 
 * @param x | The x value
 * 
 * @return square(x) such that the value outputted is safe for float arithmetic operations.
 */
float flt_safe_square(float x);

/*
 * Generates a random number between a minimum and maximum value, inclusive. 
 *
 * @param min | The minimum value
 * @param max | The maximum value
 * @param precision | The decimal precision of the minimum value and maximum value
 * 
 * @return A random number between a minimum value and a maximum value.
 */
float random_num(float min, float max, float precision);

/*
 * Randomly samples a certain number of indices within a certain range.
 *
 * @param num_samples | The number of sampled elements
 * @param num_total_data | The number of total data
 * 
 * @return an array of indices
 */
int* sample_indices(int num_samples, int num_total_data);

/*
 * Swaps two elements. 
 */
void swap(int *a, int *b);

/*
 * Randomly shuffles the elements in an array
 */
void shuffle(int *arr, int n);

/*
 * Generates the training batches to be used by the neural network for one epoch.
 *
 * @warning Batch size must be less than the total number of data points.
 *
 * @param num_data | Total number of data.
 * @param batch_size | The size of the batches.
 * @param num_batches_ref | Reference to the number of batches.
 * @param num_per_batch_ref | Reference to list storing the number data points per batch.
 * 
 * @return Randomized batches of training data.
 */
int** generate_training_batches(int num_data, int batch_size, int* num_batches_ref, int** num_per_batch_ref);

/*
 * Returns true (1) if the current number of epochs is past the maximum number of epochs.
 * 
 * @param curr_val | The current number of epochs.
 * @param lim_val | The maximum number of epochs.
 * 
 * @return 1 if true or 0 if false.
 */
int past_max_epochs(float curr_val, float lim_val);

/*
 * Returns true (1) if the difference between the current value and last value is past the 
 * minimum difference allowed between them.
 * 
 * @param last_val | The last value.
 * @param curr_val | The current value.
 * @param min_diff | The allowed minimum difference.
 * 
 * @return 1 if true or 0 if false. 
 */
int past_min_diff(float last_val, float curr_val, float min_diff); 

/*
 * Returns true (1) if the current error is past or lower than the fair error (i.e. an error that
 * is deemed acceptable).  
 * 
 * @param curr_err | The current error.
 * @param fair_err | The fair error.
 * 
 * @return 1 if true or 0 if false.
 */
int past_fair_error(float curr_err, float fair_err);

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
float mat_val(float* mat, int ncols, int i, int j);

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
void set_mat_val(float* mat, int ncols, int i, int j, float new_val);

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
int mats_equal(float* mat1, float* mat2, int nrows, int ncols);

/*
 * Flattens a given set of feature maps into a one dimensional vector. If the given set of feature maps
 * contains only one feature map (i.e. there is only one channel) then no flattening occurs, since an image
 * or feature map is already represented as a one dimensional vector in terms of this CNN implementation
 * 
 * @param feature_map_set | 
 */
float* flatten_img(void* img, int num_rows, int num_cols, int num_channels);

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
float min_vect_elem(void* vect, int num_rows, int num_cols, int num_layers); 

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
                  const char* vect_name, int vect_name_len, FILE* output_file); 

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
void disp_test_results(const char* testing_function, const char* conducting_test, void* expected, 
                       void* actual, int testing_value, FILE* output_file);

#ifdef __cplusplus
}
#endif

#endif