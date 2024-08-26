/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 * --> Contains implementations of Mean Squared Error and its derivative
 * --> Future updates:
 *          > Will contain implementation of Cross Entropy Loss and its derivative
 */

#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <math.h>

// Mean Squared Error Cost Function
float mse(float yielded_vals[], float actual_vals[], int n, int deriv_var_index) {
    if (deriv_var_index >= 0) 
        return (2 * yielded_vals[deriv_var_index] - 2 * actual_vals[deriv_var_index]) / n;

    float summation = 0;

    for (int i = 0; i < n; ++i) {
        summation += powf(yielded_vals[i] - actual_vals[i], 2);
    }

    return summation / n;
}

#endif