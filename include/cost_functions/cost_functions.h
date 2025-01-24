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
 * using a given true probability distribution and a given predicted probability distribution yielded from 
 * a neural network classification. If calculating the derivative, calculates the derivative of the expected 
 * entropy with respect to a probability from the predicted probability distribution yielded from the neural 
 * network classification.  
 * 
 * @param y_pdistro | The true probability distribution for the classification
 * @param y_hat_distro | Predicted prob distro for the classification (i.e. activated outputs from output layer)
 * @param num_classes | The number of classes in the classification.
 * @param deriv_idx | Index of the probability output taking derivative with respect to.
 * 
 * @return The cross entropy for a classification or its deriviative with respect to predicted prob distro 
 */
float multiclass_ce(float* y_pdistro, float* y_hat_pdistro, int num_classes, int deriv_idx); 

#ifdef __cplusplus
}
#endif

#endif