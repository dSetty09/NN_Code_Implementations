

#ifndef FULLY_CONN_LAYER_H
#define FULLY_CONN_LAYER_H

#include <stdlib.h>
#include "../common_definitions.h"
#include "../activation_functions/C++/nonlinear_functions.h"


/** NEURON DEFINITIONS **/

/*
 * Calculates the sum of each input vector entry multiplied by their associated weight.
 *
 * @param input_vect | A vector of inputs
 * @param assoc_weights | A vector of weights to be multiplied with each associated input
 * @param num_inputs | The number of inputs
 * 
 * @return The calculated weighted sum of each input vector entry multiplied by their associated weight
 */
float weighted_sum(float* input_vect, float* assoc_weights, float bias, int num_inputs) {
    float result = bias;

    for (int i = 0; i < num_inputs; ++i) {
        result += input_vect[i] * assoc_weights[i];
    }

    return result;
}

/*
 * Calculates the output to be produced by passing a given weighted sum as an argument to an activation function
 *
 * @param weighted_sum | The weighted sum to be processed by an activation function
 * @param activation_func | The activation function processing a given weighted sum
 * 
 * @return The output produced from passing a given weighted sum as an argument to an activation
 */
float output(float weighted_sum, one_arg_activation_function activation_func) {
    return activation_func(weighted_sum, 0);
}

/* A single neuron in a fully connected layer */
typedef struct neuron {
    float* input_vect; // The vector of inputs (i.e. feature vector taken as input from fully connected layer)
    float* assoc_weights; // The weights to be multiplied, respectively, with each input in the input vector
    float bias; // The bias for this neuron

    int num_inputs; // The number of inputs being processed and, by extension, the number of associated weights

    one_arg_activation_function activation_func; // The activation function for this neuron

    float (*weighted_sum) (float*, float*, float, int); // The function for calculating the weighted sum
    float (*output) (float, one_arg_activation_function); // The function for calculating the neuron's output
} Neuron;


/** FULLY CONNECTED LAYER DEFINITIONS **/

/*
 * Initializes the fields within a fully connected layer structure based on a given feature vector, an 
 * optional vector of weights, the specified number of neurons for this layer, the number of inputs being
 * processed by this layer, and the activation function specified to be associated with this layer.
 * 
 * FIELDS TO BE INITIALIZED:
 * neurons - List of neurons, whose weights will be init to optional weights, if given, or, otherwise randomized
 * num_neurons - The number of neurons in this layer
 * activation_func - The activation function to be used by each neuron in this layer
 * weighted_sum - The function calculating the weighted sum for each neuron
 * output - The function calculating the output for each neuron
 * 
 * @param neurons_ref | A reference to the list of neurons attribute of the layer being initialized
 * @param num_neurons_ref | A reference to the number of neurons attribute of the layer being initialized
 * @param activ_func_ref | A reference to the activation function attribute of the layer being initialized
 * @param feature_vect_input | The feature vector produced by the layer just before the layer being initialized
 * @param optional_weights | A vector of weights assigned to each connection each neuron has with each input
 * @param num_neurons | The number of neurons in the layer being initialized
 * @param num_inputs | The number of inputs in the feature vector for this layer
 * @param activation_func | The activation function to be associated with this layer
 */
void builder(Neuron** neurons_ref, int* num_neurons_ref, float* feature_vect_input, float* optional_weights, 
             int num_neurons, int num_inputs, one_arg_activation_function activation_func) {

    for (int i = 0; i < num_neurons; ++i) {
        (*neurons_ref)[i].input_vect = feature_vect_input;

        if (optional_weights) {
            (*neurons_ref)[i].assoc_weights = optional_weights;
        } else {
            for (int j = 0; j < num_inputs; ++j) {
                (*neurons_ref)[i].assoc_weights[j] = random_num(-2, 2, 3);
            }
        }

        (*neurons_ref)[i].bias = 1;
        (*neurons_ref)[i].num_inputs = num_inputs;
        (*neurons_ref)[i].activation_func = activation_func;
        (*neurons_ref)[i].weighted_sum = weighted_sum;
        (*neurons_ref)[i].output = output;
    }

    *num_neurons_ref = num_neurons;
}

/*
 * For each neuron in this layer, calculates the output to be produced by each neuron in this layer
 *
 * @param feature_vect | The feature vector being taken as input for this layer
 * @param neurons | The list of neurons in this layer
 * @param num_neurons | The number of neurons in this layer
 * 
 * @return A vector storing the outputs produced by each neuron
 */
float* exec(float* feature_vect, Neuron* neurons, int num_neurons) {
    float* output_vect = (float*) malloc(sizeof(float) * num_neurons);

    for (int i = 0; i < num_neurons; ++i) {
        output_vect[i] = neurons[i].output(neurons[i].weighted_sum(neurons[i].input_vect, neurons[i].assoc_weights, 
                                neurons[i].bias, neurons[i].num_inputs), 0);
    }

    return output_vect;
}

/*
 * Layer which either takes a flattened vector consisting of the feature maps outputted by the previous layer 
 * (if the last layer was an image processing layer) or a vector of features from the previous layer (if the last
 * layer was a fully connected layer)
 */
typedef struct fully_connected_layer {
    Neuron* neurons; // The list of neurons in this layer
    int num_neurons; // The number of neurons in this layer

    void (*build) (float**, Neuron**, int*, one_arg_activation_function*, float*, float*, int, int, 
                   one_arg_activation_function); // The builder function for this layer

    float* (*exec) (float*, Neuron*, int); // The execution function for this layer
} FullConnLayer;

#endif