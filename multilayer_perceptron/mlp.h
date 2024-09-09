/* Notes:
 * --> Contains implementation for multilayer perceptron, which is implemented as a graph containing
 *     a combination of interconnected nodes
 */

#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../common_definitions.h"


#define OUTPUT 0
#define WEIGHT 1
#define BIAS 2


/** Main struct and function definitions **/
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
    int num_layers;
    int num_weights;
    int num_total_neurons;

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

    if (deriv_index >= 0) result -= neuron.bias + 1;

    return result;
}

float neuron_output(Neuron neuron, int deriv) {
    return neuron.activation_function(weighted_sum(neuron, -1, 0), deriv);
} 

MLP* build_mlp(int num_layers, int neurons_per_layer[], one_arg_activation_function function_per_layer[], float weight_vals[]) {

    MLP* new_mlp = (MLP*) malloc(sizeof(MLP));
    new_mlp->num_layers = num_layers;
    new_mlp->num_weights = 0;
    new_mlp->num_total_neurons = 0;
    Neuron* prev_layer = NULL;
    Neuron* current_layer = new_mlp->first_layer;

    int weights_added = 0;

    new_mlp->num_first_layer_neurons = neurons_per_layer[0];

    for (int l = 0; l < num_layers; ++l) {
        current_layer = (Neuron*) malloc(sizeof(Neuron) * neurons_per_layer[l]);
        if (l == 0) new_mlp->first_layer = current_layer;

        new_mlp->num_total_neurons += neurons_per_layer[l];

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

                new_mlp->num_weights += neurons_per_layer[l - 1];

                for (int j = 0; j < neurons_in_last_layer; ++j) {
                    if (weight_vals) current_layer[i].prev_layer_weights[j] = weight_vals[weights_added++];
                    else current_layer[i].prev_layer_weights[j] = random_num(-2, 2, 3);

                    prev_layer[j].next_layer = current_layer;
                    prev_layer[j].num_in_next_layer = neurons_per_layer[l];
                }
            }

            current_layer[i].bias = 0;
            current_layer[i].activation_function = function_per_layer[l];

            current_layer[i].next_layer = NULL;
            current_layer[i].num_in_next_layer = 0;

            current_layer[i].output = NAN; // output set to NAN to indicate mlp hasn't computed anything yet
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

/* Function for visualizing an MLP */
void print_mlp(MLP* mlp, FILE* output_file) {
    int neuron_box_width = 32;
    int neuron_box_content_width = 30;
    int neuron_header_width = 9;
    int neuron_pll_field_width = 26;
    int neuron_pll_subfield_width = 30;
    int neuron_bias_field_width = 21;
    int neuron_activation_field_width = 23;
    int neuron_output_field_width = 23;

    int min_layer_length = 7;

    int num_in_curr_layer = mlp->num_first_layer_neurons;

    for (Neuron* curr_layer = mlp->first_layer; curr_layer; curr_layer = curr_layer->next_layer) {
        int* num_conns_per_neuron = (int*) malloc(sizeof(int) * num_in_curr_layer);
        for (int i = 0; i < num_in_curr_layer; ++i) num_conns_per_neuron[i] = curr_layer[i].num_in_prev_layer;
        int length_of_this_layer = min_layer_length + float_arr_max(num_conns_per_neuron, num_in_curr_layer);

        // Printing the top border for this layer
        for (int c = 0; c < neuron_box_width * num_in_curr_layer; ++c) fprintf(output_file, "-");
        fprintf(output_file, "\n|");

        // Printing headers for each respective neuron in the layer
        for (int i = 0; i < num_in_curr_layer; ++i) {
            fprintf(output_file, "Neuron %d:", i);
            for (int c = 0; c < neuron_box_content_width - neuron_header_width; ++c) fprintf(output_file, " ");
            fprintf(output_file, "|");
        }

        fprintf(output_file, "\n|");

        // Printing the previous layer field for each respective neuron
        for (int i = 0; i < num_in_curr_layer; ++i) {
            fprintf(output_file, "    -> Prev Layer Links: %d", curr_layer[i].num_in_prev_layer);
            for (int c = 0; c < neuron_box_content_width - neuron_pll_field_width; ++c) fprintf(output_file, " ");
            fprintf(output_file, "|");
        }

        // Printing each connection and their associated weight for each respective neuron
        for (int l = length_of_this_layer - min_layer_length; l >= 0; --l) {
            fprintf(output_file, "\n|");

            for (int i = 0; i < num_in_curr_layer; ++i) {
                int num_spaces = neuron_box_content_width;

                int w_idx = abs(l - (length_of_this_layer - min_layer_length));
                if (w_idx < curr_layer[i].num_in_prev_layer) {
                    float rounded_weight = round_to_place(curr_layer[i].prev_layer_weights[w_idx], 3);
                    fprintf(output_file, "        [%d: Weight = %f]", w_idx, rounded_weight);

                    num_spaces -= neuron_pll_subfield_width;
                }


                for (int c = 0; c < num_spaces; ++c) fprintf(output_file, " ");
                fprintf(output_file, "|");
            }
        }

        fprintf(output_file, "\n|");

        // Printing the bias field for each respective neuron
        for (int i = 0; i < num_in_curr_layer; ++i) {
            fprintf(output_file, "    -> Bias: %f", round_to_place(curr_layer[i].bias, 0));
            for (int c = 0; c < neuron_box_content_width - neuron_bias_field_width; ++c) fprintf(output_file, " ");
            fprintf(output_file, "|");
        }

        fprintf(output_file, "\n|");

        // Printing the output field for each respective neuron
        // TODO: (if have time) change this part so that adaptable to various output field widths
        for (int i = 0; i < num_in_curr_layer; ++i) {
            fprintf(output_file, "    -> Output: %f", round_to_place(curr_layer[i].output, 3));
            for (int c = 0; c < neuron_box_content_width - neuron_output_field_width; ++c) fprintf(output_file, " ");
            fprintf(output_file, "|");
        }

        fprintf(output_file, "\n");

        // Printing the bottom border for this layer
        for (int c = 0; c < neuron_box_width * num_in_curr_layer; ++c) fprintf(output_file, "-");
        fprintf(output_file, "\n|");

        fprintf(output_file, "\n\n\n");

        num_in_curr_layer = curr_layer->num_in_next_layer;
    }
}

/* Emulates backward pass functionality of an MLP */
void adjust_weights_and_biases(MLP* mlp, cost_function loss_function, float actual_outputs[]) {
    float* stored_weights_adjustments = (float*) malloc(sizeof(float) * mlp->num_weights);
    int swa_i = 0;

    float* stored_bias_adjustments = (float*) malloc(sizeof(float) * mlp->num_total_neurons);
    int sba_i = 0;

    float* last_layer_derivs = (float*) malloc(sizeof(float) * mlp->num_output_layer_neurons);
    int num_in_last_layer = mlp->num_output_layer_neurons;
    
    for (int i = 0; i < mlp->num_output_layer_neurons; ++i) last_layer_derivs[i] = loss_function(mlp->outputs, actual_outputs, mlp->num_output_layer_neurons, i);

    int num_in_curr_layer = mlp->output_layer->num_in_prev_layer;

    for (Neuron* current_layer = mlp->output_layer->prev_layer; current_layer; current_layer = current_layer->prev_layer) {
        float* cost_over_output_derivs = (float*) calloc(num_in_curr_layer, sizeof(float));

        for (int i = 0; i < num_in_curr_layer; ++i) {
            for (int j = 0; j < num_in_last_layer; ++j) {
                float jth_output_to_wsum_deriv = current_layer[i].next_layer[j].activation_function(weighted_sum(current_layer[i].next_layer[j], -1, 0), 1);
                float jth_wsum_to_ith_output_deriv = weighted_sum(current_layer[i].next_layer[j], i, OUTPUT);
                float cost_deriv_addend = jth_wsum_to_ith_output_deriv * jth_output_to_wsum_deriv * last_layer_derivs[j];
                if (!isnan(cost_deriv_addend)) cost_over_output_derivs[i] += cost_deriv_addend;
            }
        }

        for (int j = 0; j < num_in_last_layer; ++j) {
            for (int i = 0; i < num_in_curr_layer; ++i) {
                float jth_output_to_wsum_deriv = current_layer[i].next_layer[j].activation_function(weighted_sum(current_layer[i].next_layer[j], -1, 0), 1);

                float jth_wsum_to_weight_ij_deriv = weighted_sum(current_layer[i].next_layer[j], i, WEIGHT);
                float weight_deriv = last_layer_derivs[j] * jth_output_to_wsum_deriv * jth_wsum_to_weight_ij_deriv;
                stored_weights_adjustments[swa_i++] = (isnan(weight_deriv)) ? 0 : weight_deriv; // weight derivative being equal to nan only occurs in case where 0 multiplied by infinity

                float jth_wsum_to_bias_deriv = weighted_sum(current_layer[i].next_layer[j], i, BIAS);
                float bias_deriv = last_layer_derivs[j] * jth_output_to_wsum_deriv * jth_wsum_to_bias_deriv;
                stored_bias_adjustments[sba_i++] = (isnan(bias_deriv)) ? 0 : bias_deriv; // weight derivative being equal to nan only occurs in case where 0 multiplied by infinity
            }
        }

        free(last_layer_derivs);
        last_layer_derivs = cost_over_output_derivs;

        num_in_last_layer = num_in_curr_layer;
        num_in_curr_layer = current_layer->num_in_prev_layer;
    }

    swa_i = 0;
    sba_i = 0;

    for (Neuron* current_layer = mlp->output_layer; current_layer->prev_layer; current_layer = current_layer->prev_layer) {
        int num_in_current_layer = current_layer->prev_layer->num_in_next_layer;
        int num_in_prev_layer = current_layer->num_in_prev_layer;

        for (int j = 0; j < num_in_current_layer; ++j) {
            for (int i = 0; i < num_in_prev_layer; ++i) {
                current_layer[j].prev_layer_weights[i] -= stored_weights_adjustments[swa_i++];
            }

            current_layer[j].bias -= stored_bias_adjustments[sba_i++];
        }
    }
}

void decommision_mlp(MLP* mlp) {

}

#endif