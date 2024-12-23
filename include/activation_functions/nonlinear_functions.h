#ifndef NONLINEAR_FUNCTIONS_H
#define NONLINEAR_FUNCTIONS_H

#include "../neural_net_ops/neural_net_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CONSTANTS USED IN ACTIVATION FUNCTIONS */
static const float SIG_MIN_INPUT = -88.72283;
static const float SIG_MAX_INPUT = 88.72283;

static const float TANH_MIN_INPUT = -44;
static const float TANH_MAX_INPUT = 44;

// Sigmoid Activation Function
// --> Used for models where have to predict probability as output
float sigmoid(float x, int deriv); 

// Hyperbolic Tangent (tanh) Function
// --> Alternative to sigmoid function, except range is from
//     -1 to 1
float hyperbolic_tangent(float x, int deriv); 

// Step Function
// --> Used in binary classification
// --> Not useful for neural network learning as derivative
//     is equal to 0, except where x = 0 (as the derivative
//     doesn't exist at this point)
float step(float x, int deriv);

// Rectified Linear Unit (ReLU) Activation Function
// --> Similar to identity function except values less than
//     0 are mapped to 0
float relu(float x, int deriv);
    
// Leaky Rectified Linear Unit Activation Function
// --> Like ReLU except designed to handle the case where
//     weighted sum is less than 0. 
float leaky_relu(float x, int deriv);
    
// SoftPlus Activation Function
// --> Smoother approximation of ReLU
float softplus(float x, int deriv); 

// SoftMax Activation Function
// --> Applied in multi-class classification and used
//     to determine the probability of a weighted sum value
//     being the possible true output for a given input
// ---> There are n classes
float softmax(float* z, int i, unsigned int n, int deriv_idx);

#ifdef __cplusplus
}
#endif

#endif
