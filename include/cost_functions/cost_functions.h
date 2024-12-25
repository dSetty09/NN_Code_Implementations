#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "../neural_net_ops/neural_net_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Calculates the squared error for a specific data point. 
 * If it is specified to take the derivative , then calculates the 
 * derivative of the mean squared error with respect to y_hat for that specific point.
 * 
 * @param y_m | True output for the specified data point.  
 * @param y_hat_m | Predicted output for the specified data point.
 * @param deriv | Flag indicating whether taking derivative.
 * 
 * @return The mean squared error or derivative with respect to specific output neuron. 
 */
float se(float y_m, float y_hat_m, int deriv); 

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
float cross_entropy(float* yielded_distro, float* true_distro, int n, int deriving_output_index); 

#ifdef __cplusplus
}
#endif

#endif