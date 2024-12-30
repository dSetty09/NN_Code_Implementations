#ifndef UNIT_H
#define UNIT_H

#include "../neural_net_ops/neural_net_ops.h"

#ifdef __cplusplus
extern "C" {
#endif


/* DEFINING STRUCTS FOR KEY NEURAL NETWORK UNITS */

/*
 * A neuron in a dense layer within a neural network.
 */
typedef struct neuron_node {
    int W; // the number of weights 

    float* w; // the vector of weights connected to each neuron in the preceding layer
    float b; // the bias term

    float* delta_w; // the gradient of the weights connected to this neuron
    float delta_a; // the gradient of the activation for this neuron 
    float delta_b; // the gradient of the bias for this neuron

    simple_act_func act_func; // the activation function for the neuron, NULL if softmax defined
    softmax_act_func soft_act_func; // softmax activation function for neuron, NULL if act_func defined

    float output; // the output of the neuron
} Neuron;

typedef struct kernel {
    float placeholder; // ITS A THING!!! :D
} Kernel;


/* DEFINING FUNCTIONS FOR NEURON OPERATIONS */

/*
 * Calculates the weighted sum of inputs for a neuron with a specific set of weights or calculates
 * derivative with respect to a specific weight or bias for that neuron.
 * 
 * @param neuron | The neuron for which the weighted sum is calculated.
 * @param x | A vector of inputs received from the preceding layer. 
 * @param deriv_wx_idx | The index of the weight or input with respect to which derivative being taken of.  
 * @param deriv_w | Flag indicating whether taking derivative with respect to a weight or not.
 * @param deriv_x | Flag indicating whether taking derivative with respect to an input or not. 
 * @param deriv_b | Flag indicating whether taking derivative with respect to bias or not.
 * 
 * @return The calculated weighted sum.
 */ 
float wsum(Neuron* neuron, float* x, int deriv_wx_idx, int deriv_w, int deriv_x, int deriv_b); 

/*
 * Updates the weights, inputs, and bias gradients for the referenced neuron in relation to the result
 * of the cost function result from the forward pass of the neural network. 
 * 
 * @param neuron | The neuron for which the gradients are being calculated.
 * @param x | A vector of inputs received from the preceding layer.
 * @param cost_act_derivs | The derivs of the cost function w/ respect to activation of this neuron. 
 * 
 * @return The derivatives of the cost function with respect to each input passed to this neuron.
 */
float* update_gradients(Neuron* neuron, float* x, float* cost_act_derivs, int num_derivs); 


#ifdef __cplusplus
}
#endif

#endif