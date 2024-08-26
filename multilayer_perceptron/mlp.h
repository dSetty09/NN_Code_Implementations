/* Notes:
 * --> Contains implementation for multilayer perceptron, which is implemented as a graph containing
 *     a combination of interconnected nodes
 */

#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <time.h>
#include "../common_typedefs.h"


float generate_random_weight() {
    srand(time(NULL));
    return rand() % 5;
}


struct Neuron {
    Neuron** prev_layer_connections;
    float* prev_layer_weights;
    int num_prev_layer_connections;

    float bias;
    one_arg_activation_function activation_function;

    Neuron** next_layer_connections;
    int num_next_layer_connections;

    float output;
};

struct MultilayerPerceptron {
    Neuron** first_layer;
    int num_first_layer_neurons;
};

float neuron_output(Neuron* neuron) {
    float weighted_sum = neuron->bias;

    for (int i = 0; i < neuron->num_prev_layer_connections; ++i) {
        weighted_sum += neuron->prev_layer_connections[i]->output * neuron->prev_layer_weights[i];
    }

    return neuron->activation_function(weighted_sum, 0); //  second argument, 0, implies not taking derivative
} 

MultilayerPerceptron* build_mlp(int num_layers, int* neurons_per_layer, 
                                one_arg_activation_function* function_per_layer, 
                                cost_function cost_func) {

    MultilayerPerceptron* new_mlp = (MultilayerPerceptron*) sizeof(MultilayerPerceptron);
    Neuron** prev_layer = NULL;
    Neuron** current_layer = new_mlp->first_layer;

    new_mlp->num_first_layer_neurons = neurons_per_layer[0];

    for (int i = 0; i < num_layers; ++i) {
        current_layer = (Neuron**) malloc(sizeof(Neuron*) * neurons_per_layer[i]);

        for (int j = 0; j < neurons_per_layer[i]; ++j) {
            current_layer[j] = (Neuron*) malloc(sizeof(Neuron));

            if (!prev_layer) {
                current_layer[j]->prev_layer_connections = NULL;
                current_layer[j]->prev_layer_weights = NULL;
                current_layer[j]->num_prev_layer_connections = 0;
            } else {
                for (int k = 0; k < neurons_per_layer[i - 1]; ++k) {
                    prev_layer[k]->next_layer_connections[j] = current_layer[j];
                    
                    current_layer[j]->prev_layer_connections[k] = prev_layer[k];
                    current_layer[j]->prev_layer_weights[k] = generate_random_weight();
                }

                current_layer[j]->num_prev_layer_connections = neurons_per_layer[i - 1];
            }

            current_layer[j]->bias = 0;
            current_layer[j]->activation_function = function_per_layer[i];
            current_layer[j]->next_layer_connections = NULL;
            current_layer[i]->num_next_layer_connections = 0;

            current_layer[j]->output = 0; // neuron "switched off" when initialized
        }

        prev_layer = current_layer;
    }

    return new_mlp;
}

float* compute_mlp_output(MultilayerPerceptron* mlp, int* inputs) {
    int num_outputs = 0;
    float* outputs = NULL;

    Neuron** curr_layer = mlp->first_layer;
    int num_neurons_in_curr_layer = mlp->num_first_layer_neurons;
    int on_first_layer = 1;

    while (curr_layer != NULL) {
        int num_neurons_in_next_layer = curr_layer[0]->num_next_layer_connections;

        for (int i = 0; i < num_neurons_in_curr_layer; ++i) {
            if (on_first_layer) {
                curr_layer[i]->output = curr_layer[i]->activation_function(inputs[i], 0);
            } else {
                curr_layer[i]->output = neuron_output(curr_layer[i]);
            }
        }

        if (num_neurons_in_next_layer == 0) {
            num_outputs = num_neurons_in_curr_layer;

            for (int i = 0; i < num_outputs; ++i) {
                outputs[i] = curr_layer[i]->output;
            }
        }

        curr_layer = curr_layer[0]->next_layer_connections;
        num_neurons_in_curr_layer = num_neurons_in_next_layer;

        on_first_layer = 0;
    }
    
    return outputs;
}

void decommision_mlp(MultilayerPerceptron* mlp) {

}

#endif