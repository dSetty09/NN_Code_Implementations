/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 * --> Contains implementations of Mean Squared Error and its derivative
 * --> Future updates:
 *          > Will contain implementation of Cross Entropy Loss and its derivative
 */

#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <math.h>
#include "common_definitions.h"

/*
 * Calculates the mean squared error between a set of values yielded from a neural network versus 
 * the true set of values. If the deriving output index is greater than or equal to 0, calculates the
 * derivative of the mean squared error with respect to the yielded value from the neuron associated
 * with the specified deriving output index.
 *
 * @param yielded_vals | The values yielded from the neurons in the last layer of a neural network
 * @param actual_vals | The true values that should have been produced from the last layer
 * @param n | The number of values being compared
 * @param deriving_output_index | The index associated with the neuron whose output used to calc derivative
 * 
 * @return The mean squared error or its derivative
 */
float mse(float* yielded_vals, float* actual_vals, int n, int deriving_output_index) {
    if (deriving_output_index >= 0) 
        return (2 * yielded_vals[deriving_output_index] - 2 * actual_vals[deriving_output_index]) / n;

    float summation = 0;

    for (int i = 0; i < n; ++i) {
        summation += bp_safe_square(yielded_vals[i] - actual_vals[i], 0);
    }

    return summation / n;
}

/*
 * Calculates the expected entropy, or uncertainty, of the classification made from the neural network
 * using the distribution it yielded, rather than the true distribution associated with the classification.
 * The true distribution makes the correct classification. If the deriving_output_index is greater than
 * or equal to 0, calculates the derivative of the expected entropy with respect to the yielded probability 
 * from the neuron associated with the specified deriving output index.
 * 
 * @param yielded_distro | The probability distribution yielded from a neural network classification
 * @param true_distro | The true probability distribution for the classification
 * @param n | The number of outcomes in the probability distribution
 * @param deriving_output_index | The index associated with the probability with which deriving in respect to
 * 
 * @return The cross entropy for a classification or its deriviative with respect to a certain probability
 */
float cross_entropy(float* yielded_distro, float* true_distro, int n, int deriving_output_index) {
    if (deriving_output_index >= 0) {
        return -(true_distro[deriving_output_index] / (yielded_distro[deriving_output_index] + NEAR_ZERO));
    }

    float yielded_distro_entropy = 0;

    for (int i = 0; i < n; ++i) {
        yielded_distro_entropy += true_distro[i] * bp_safe_log(yielded_distro[i], 0);
    }

    return -yielded_distro_entropy;
}

#endif