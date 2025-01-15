#include "../../include/ann/dense_layer.h"

/* FUNCTIONS FOR HANDLING DENSE LAYERS */

DenseLayer init_dense_layer(int num_neurons, ...) {
    DenseLayer ret = {.neurons=(NeuronNode*) malloc(sizeof(NeuronNode) * num_neurons), .num_neurons=num_neurons};

    va_list neuron_specs; 
    va_start(neuron_specs, num_neurons);

    for (int i = 0; i < num_neurons; ++i) ret.neurons[i] = init_neuron(va_arg(neuron_specs, Neuron));

    va_end(neuron_specs);

    return ret;
}

void del_dense_layer(DenseLayer dl) {
    for (int i = 0; i < dl.num_neurons; ++i) del_neuron(dl.neurons[i]);
}


/* FUNCTIONS FOR DENSE LAYER OPERATIONS */

void start_forward_pass(DenseLayer* dl, float** outputs_ref) {
    *outputs_ref = (float*) malloc(sizeof(float) * dl->num_neurons);
}

void conduct_forward_pass(NeuronNode* neurons, int n, float* x, int softmax_used, float* outputs) {
    float* wsums = (float*) malloc(sizeof(float) * n);

    for (int i = 0; i < n; ++i) wsums[i] = wsum(neurons + i, x, NO_DERIV, FALSE, FALSE, FALSE);

    for (int i = 0; i < n; ++i) 
        outputs[i] = (softmax_used) ? ((softmax_act_func) neurons[i].act_func)(wsums, i, n, NO_DERIV) : 
                                      ((simple_act_func) neurons[i].act_func)(wsums[i], FALSE);

    free(wsums);
}

void end_forward_pass(float* outputs) {
    free(outputs);
}

void conduct_back_prop(DenseLayer* dl) {

}
