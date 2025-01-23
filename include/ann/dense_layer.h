#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H 

#include "../unit/unit.h"

#ifdef __cplusplus
extern "C" {
#endif


/* DENSE LAYER STRUCT */

/*
 * Represents a dense layer in a neural network.
 *
 * @attention This struct is also used for facilitating the creation of a new dense layer by treating
 * this struct as the set of parameters for a new dense layer
 * 
 */
typedef struct dl {
    NeuronNode* neurons; // the array of neuron nodes in this layer
    int num_neurons; // the number of neurons in this layer 
} DenseLayer;


/* FUNCTION FOR HANDLING DENSE LAYER */

/*
 * Creates a new dense layer according to the given dense layer parameters.
 *
 * @params num_neurons | The number of neurons to be in the newly created dense layer
 * @params ... | A variable number of neuron specifications
 * 
 * @warning The behavior of this function is undefined when at least one argument, that is not a "Neuron" or 
 * "neuron_spec" is passed as a variable argument
 * 
 * @return A reference to the newly created dense layer.
 */
DenseLayer init_dense_layer(int num_neurons, ...);

/*
 * Deletes the memory associated with dense layer.
 *
 * @param dl | Reference to dense layer whose memory is being deleted.
 */
void del_dense_layer(DenseLayer* dl);


/* FUNCTIONS FOR DENSE LAYER OPERATIONS */

/*
 * Initializes the memory for data structures needed to complete the forward pass operation for a specific 
 * dense layer.
 *
 * @param dl | The dense layer of concern
 * @param outputs_ref | A reference to the outputs array.
 */
void start_forward_pass(DenseLayer* dl, float** outputs_ref);

/*
 * Conducts a forward pass operation for a specific dense layer. More specifically, the outputs for each neuron
 * in the layer are calculated and stored in a certain output array.
 *
 * @param neurons | The neurons that reside within the dense layer of concern.
 * @param n | The number of neurons within the dense layer of concern.
 * @param x | The array of inputs given to this layer. 
 * @param softmax_used | A flag indicating whether the neurons in this layer use the softmax activation.
 * @param outputs_ref | The outputs array.
 */
void conduct_forward_pass(NeuronNode* neurons, int n, float* x, int softmax_used, float* outputs);

/*
 * Frees the memory for data structures needed to complete the forward pass operation.
 *
 * @param outputs | The outputs array.
 */
void end_forward_pass(float* outputs);

/*
 * DESCRIPTION DUE SOON
 */
void conduct_back_prop(DenseLayer* dl);

#ifdef __cplusplus
}
#endif

#endif