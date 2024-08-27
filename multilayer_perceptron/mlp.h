/* Notes:
 * --> Contains implementation for multilayer perceptron, which is implemented as a graph containing
 *     a combination of interconnected nodes
 */

#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <time.h>

#include "../common_definitions.h"


float generate_random_weight() {
    srand(time(NULL));
    return (rand()% 2 == 1) ? rand() % 5 : -(rand() % 5);
}

typedef struct NeuronNode {
    struct NeuronNode* prev_layer;
    float* prev_layer_weights;
    int num_in_prev_layer;

    float bias;
    one_arg_activation_function activation_function;

    struct NeuronNode* next_layer;
    int num_in_next_layer;

    float output;
} Neuron;

typedef struct MultilayerPerceptron {
    Neuron* first_layer;
    int num_first_layer_neurons;
} MLP;

float weighted_sum(Neuron neuron, int deriv_output_index) {
    if (deriv_output_index >= 0) return neuron.prev_layer_weights[deriv_output_index];

    float result = neuron.bias;

    for (int i = 0; i < neuron.num_in_prev_layer; ++i) {
        result += neuron.prev_layer[i].output * neuron.prev_layer_weights[i];
    }

    return result;
}

float neuron_output(Neuron neuron) {
    return neuron.activation_function(weighted_sum(neuron, -1), 0); //  second argument, 0, implies not taking derivative
} 

MLP* build_mlp(int num_layers, int neurons_per_layer[], 
                                one_arg_activation_function function_per_layer[], 
                                cost_function cost_func, float weight_vals[]) {

    MLP* new_mlp = (MLP*) malloc(sizeof(MLP));
    Neuron* prev_layer = NULL;
    Neuron* current_layer = new_mlp->first_layer;

    int weights_added = 0;

    new_mlp->num_first_layer_neurons = neurons_per_layer[0];

    for (int l = 0; l < num_layers; ++l) {
        current_layer = (Neuron*) malloc(sizeof(Neuron) * neurons_per_layer[l]);
        if (l == 0) new_mlp->first_layer = current_layer;

        for (int i = 0; i < neurons_per_layer[l]; ++i) {
            if (!prev_layer) {
                current_layer[i].prev_layer = NULL;
                current_layer[i].prev_layer_weights = NULL;
                current_layer[i].num_in_prev_layer = 0;
            } else {
                current_layer[i].prev_layer = prev_layer;
                current_layer[i].prev_layer_weights = (float*) malloc(sizeof(float) * neurons_per_layer[l - 1]);
                current_layer[i].num_in_prev_layer = neurons_per_layer[l - 1];

                int neurons_in_last_layer = neurons_per_layer[l - 1];

                for (int j = 0; j < neurons_in_last_layer; ++j) {
                    if (weight_vals) current_layer[i].prev_layer_weights[j] = weight_vals[weights_added++];
                    else current_layer[i].prev_layer_weights[j] = generate_random_weight();

                    prev_layer[j].next_layer = current_layer;
                    prev_layer[j].num_in_next_layer = neurons_per_layer[l];
                }
            }

            current_layer[i].bias = 0;
            current_layer[i].activation_function = function_per_layer[l];

            current_layer[i].next_layer = NULL;
            current_layer[i].num_in_next_layer = 0;

            current_layer[i].output = 0; // neuron "switched off" when initialized
        }

        prev_layer = current_layer;
    }

    return new_mlp;
}

float* compute_mlp_output(MLP* mlp, float inputs[]) {
    int num_outputs = 0;
    float* outputs = NULL;

    Neuron* curr_layer = mlp->first_layer;
    int num_neurons_in_curr_layer = mlp->num_first_layer_neurons;
    int on_first_layer = 1;

    while (curr_layer != NULL) {
        int num_neurons_in_next_layer = curr_layer[0].num_in_next_layer;

        for (int i = 0; i < num_neurons_in_curr_layer; ++i) {
            if (on_first_layer) {
                curr_layer[i].output = inputs[i];
            } else {
                curr_layer[i].output = neuron_output(curr_layer[i]);
            }
        }

        if (num_neurons_in_next_layer == 0) {
            num_outputs = num_neurons_in_curr_layer;

            outputs = (float*) malloc(sizeof(float) * num_outputs);

            for (int i = 0; i < num_outputs; ++i) {
                outputs[i] = curr_layer[i].output;
            }
        }

        curr_layer = curr_layer[0].next_layer;
        num_neurons_in_curr_layer = num_neurons_in_next_layer;

        on_first_layer = 0;
    }
    
    return outputs;
}

void decommision_mlp(MLP* mlp) {

}

#endif