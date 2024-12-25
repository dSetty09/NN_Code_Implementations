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
 * using the predicted probability distribution yielded from the neural network classification.
 * If calculating the derivative, calculates the derivative of the expected entropy with respect to the 
 * predicted probability distribution yielded from the neural network classification.  
 * 
 * @param y_hat_distro | The predicted probability distribution for the classification
 * @param true_cls | The true classification that the neural network was supposed to output 
 * @param deriv | Flag indicating whether taking derivative.
 * 
 * @return The cross entropy for a classification or its deriviative with respect to predicted prob distro 
 */
float multiclass_ce(float* y_hat_pdistro, int true_cls, int deriv); 

#ifdef __cplusplus
}
#endif

#endif