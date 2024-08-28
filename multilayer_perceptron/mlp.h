/* Notes:
 * --> Contains implementation for multilayer perceptron, which is implemented as a graph containing
 *     a combination of interconnected nodes
 */

#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <time.h>

#include "../common_definitions.h"


#define OUTPUT 0
#define WEIGHT 1
#define BIAS 2


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

    Neuron* output_layer;
    int num_output_layer_neurons;

    float* outputs;
} MLP;

float weighted_sum(Neuron neuron, int deriv_index, int deriving_variable) {
    if (deriv_index >= 0) {
        if (deriving_variable == OUTPUT) return neuron.prev_layer_weights[deriv_index];
        if (deriving_variable == WEIGHT) return neuron.prev_layer[deriv_index].output;
    }

    float result = neuron.bias;

    for (int i = 0; i < neuron.num_in_prev_layer; ++i) {
        result += neuron.prev_layer[i].output * neuron.prev_layer_weights[i];
    }

    if (deriv_index >= 0) result -= neuron.bias;

    return result;
}

float neuron_output(Neuron neuron, int deriv) {
    return neuron.activation_function(weighted_sum(neuron, -1, 0), deriv);
} 

MLP* build_mlp(int num_layers, int neurons_per_layer[], 
                                one_arg_activation_function function_per_layer[],  float weight_vals[]) {

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

    new_mlp->output_layer = current_layer;
    new_mlp->num_output_layer_neurons = new_mlp->output_layer->prev_layer->num_in_next_layer;

    return new_mlp;
}

/* Emulates forward-pass functionality of an MLP */
void compute_mlp_output(MLP* mlp, float inputs[]) {
    int num_outputs = 0;
    float* outputs = NULL;

    Neuron* curr_layer = mlp->first_layer;
    int num_neurons_in_curr_layer = mlp->num_first_layer_neurons;
    int on_first_layer = 1;

    while (curr_layer != NULL) {
        int num_neurons_in_next_layer = curr_layer->num_in_next_layer;

        for (int i = 0; i < num_neurons_in_curr_layer; ++i) {
            if (on_first_layer) {
                curr_layer[i].output = inputs[i];
            } else {
                curr_layer[i].output = neuron_output(curr_layer[i], 0);
            }
        }

        if (num_neurons_in_next_layer == 0) {
            num_outputs = num_neurons_in_curr_layer;

            outputs = (float*) malloc(sizeof(float) * num_outputs);

            for (int i = 0; i < num_outputs; ++i) {
                outputs[i] = curr_layer[i].output;
            }
        }

        curr_layer = curr_layer->next_layer;
        num_neurons_in_curr_layer = num_neurons_in_next_layer;

        on_first_layer = 0;
    }
    
    mlp->outputs = outputs;
}

/* Emulates backward pass functionality of an MLP */
void adjust_weights_and_biases(MLP* mlp, cost_function loss_function, float actual_outputs[]) {
    float* last_layer_derivs = (float*) malloc(sizeof(float) * mlp->num_output_layer_neurons);
    int num_in_last_layer = mlp->num_output_layer_neurons;
    
    for (int i = 0; i < mlp->num_output_layer_neurons; ++i) last_layer_derivs[i] = loss_function(mlp->outputs, actual_outputs, mlp->num_output_layer_neurons, i);

    int num_in_curr_layer = mlp->output_layer->num_in_prev_layer;

    for (Neuron* current_layer = mlp->output_layer->prev_layer; current_layer; current_layer = current_layer->prev_layer) {
        float* cost_over_output_derivs = (float*) calloc(num_in_curr_layer, sizeof(float));

        for (int i = 0; i < num_in_curr_layer; ++i) {
            for (int j = 0; j < num_in_last_layer; ++j) {
                float jth_output_to_wsum_deriv = current_layer[i].next_layer[j].activation_function(weighted_sum(current_layer->next_layer[j], -1, 0), 1);

                float jth_wsum_to_weight_ij_deriv = weighted_sum(current_layer[i].next_layer[j], i, WEIGHT);
                current_layer[i].next_layer[j].prev_layer_weights[i] += jth_wsum_to_weight_ij_deriv;

                float jth_wsum_to_bias_deriv = weighted_sum(current_layer[i].next_layer[j], i, BIAS);
                current_layer[i].next_layer[j].bias += jth_wsum_to_bias_deriv;

                float jth_wsum_to_ith_output_deriv = weighted_sum(current_layer[i].next_layer[j], i, OUTPUT);
                cost_over_output_derivs[i] += jth_wsum_to_ith_output_deriv * jth_output_to_wsum_deriv * last_layer_derivs[j];
            }
        }

        free(last_layer_derivs);
        last_layer_derivs = cost_over_output_derivs;
    }
}

void decommision_mlp(MLP* mlp) {

}

#endif