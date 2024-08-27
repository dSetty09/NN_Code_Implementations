/* Notes:
 * This file contains common type definitions used across all C files in all folders
 */

#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#define TRUE 1
#define FALSE 0

typedef float (*one_arg_activation_function) (float, int);
typedef float (*cost_function) (float[], float[], int, int);

// Function for rounding to a certain number of decimal places
float round_to_place(float val, float place) {
    float multiple = powf(10, place);
    return round(val * multiple) / multiple;
}

#endif