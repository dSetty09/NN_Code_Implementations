/* Notes:
 * This file contains common type definitions used across all C files in all folders
 */

typedef float (*one_arg_activation_function) (float, int);
typedef float (*cost_function) (float[], float[], int, int);