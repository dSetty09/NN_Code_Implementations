#ifndef UNIT_H
#define UNIT_H

#include "../neural_net_ops/neural_net_ops.h"

#ifdef __cplusplus
extern "C" {
#endif


/* USEFUL MACROS FOR FACILITATING NEURON CREATION */

/*
 * Expands to a compound array literal, converting the given arguments into a float array
 */
#define WEIGHTS(...) (float[]) {__VA_ARGS__}


/* DEFINING STRUCTS FOR KEY NEURAL NETWORK UNITS */

/*
 * A neuron in a dense layer within a neural network.
 */
typedef struct neuron_node {
    float* w; // the vector of weights connected to each neuron in the preceding layer
    int num_weights; // the number of weights 

    float b; // the bias term

    float* delta_w; // the gradient of the weights connected to this neuron
    float deriv_a; // the derivative of the cost with respect to the activation for this neuron 
    float delta_b; // the gradient of the bias for this neuron

    void* act_func; // the activation function for this neuron

    float output; // the output of the neuron
} Neuron;

typedef struct kernel {
    float placeholder; // ITS A THING!!! :D
} Kernel;


/* DEFINING STRUCTS FOR MAKING NEURON CREATION PARAMETERS MORE READABLE */

/*
 * Parameters for constructing a neuron.
 *
 * Different ways to declare parameters:
 * 
 * - neuron = _neuron({.act_func=(void*) relu, .num_weights=5, .bias=0.877, WEIGHTS(0.632, 0.571, 0.991)});
 * - NeuronParams = {.act_func=(void*) relu, .num_weights=5, .bias=0.877, WEIGHTS(0.632, 0.571, 0.991)}; 
 * 
 */
typedef struct params {
    void* act_func; // activation function
    int num_weights; // number of weights

    float bias; // bias parameter
    float* weights; // weights optional keyword parameter
} NeuronParams;


/* DEFINING FUNCTION FOR CREATING NEURONS */

/*
 * Creates a neuron given a set of neuron parameters. 
 *
 * @param params | Parameters for creating the neuron.
 * 
 * @return A reference to the newly created neuron object.
 */
Neuron* _neuron(NeuronParams params);


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
 * @param num_derivs | The number of cost to activation derivatives passed to this function.
 * 
 * @return The derivatives of the cost function with respect to each input passed to this neuron.
 */
float* update_gradients(Neuron* neuron, float* x, float* cost_act_derivs, int num_derivs); 

/*
 * Similar to the above function, except the neuron for which the gradients are being updated is a 
 * neuron that has the softmax activation function. As a result, this function has an additional parameter
 * which represents the weighted sums across the layer this neuron resides in. 
 * 
 * @warning Because the softmax is intended to be used for neurons in the output layer, the referenced 
 * neuron should reside in the output layer of a neural network classifier.
 * 
 * @param neuron | The neuron for which the gradients are being calculated.
 * @param x | A vector of inputs received from the preceding layer.
 * @param cost_outputs_derivs | The derivs of the cost function w/ respect to each output in this layer. 
 * @param num_outputs | The number of cost to output derivatives passed to this function.
 * @param out_z | The weighted sums across the output layer this neuron resides in.
 */
float* update_gradients_sm(Neuron* neuron, float* x, float* cost_outputs_derivs, int num_outputs, float* out_z);


#ifdef __cplusplus
}
#endif

#endif