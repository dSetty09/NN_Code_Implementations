#include "../../include/unit/unit.h"


/* NEURON CREATION FUNCTIONS */

Neuron* _neuron(NeuronParams params) {
    Neuron* ret = (Neuron*) malloc(sizeof(Neuron));

    ret->act_func = NULL; ret->act_func = params.act_func;
    ret->num_weights = params.num_weights;

    ret->b = params.bias;
    ret->w = NULL; ret->w = (float*) malloc(sizeof(float) * ret->num_weights);

    for (int i = 0; i < ret->num_weights; ++i) 
        ret->w[i] = (params.weights) ? params.weights[i] : random_num(0, 1, 5);


    ret->delta_b = 0;
    ret->delta_w = NULL; ret->delta_w = (float*) calloc(ret->num_weights, sizeof(float)); 

    ret->deriv_a = 0;

    ret->output = 0;

    return ret;
}


/* NEURON OPERATION FUNCTIONS */

float wsum(Neuron* neuron, float* x, int deriv_wx_idx, int deriv_w, int deriv_x, int deriv_b) {
    if (deriv_b) return 1;
    if (deriv_w) return x[deriv_wx_idx];
    if (deriv_x) return neuron->w[deriv_wx_idx];

    float sum = neuron->b;

    for (int i = 0; i < neuron->num_weights; ++i) sum += neuron->w[i] * x[i];

    return sum;
}

float* update_gradients(Neuron* neuron, float* x, float* cost_act_derivs, int num_derivs) {
    return NULL;
}

float* update_gradients_sm(Neuron* neuron, float* x, float* cost_outputs_derivs, int num_outputs, float* out_z) {
    return NULL;
}