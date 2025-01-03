#include "../../include/unit/unit.h"
#include "stdio.h"

float wsum(Neuron* neuron, float* x, int deriv_wx_idx, int deriv_w, int deriv_x, int deriv_b) {
    if (deriv_b) return 1;
    if (deriv_w) return x[deriv_wx_idx];
    if (deriv_x) return neuron->w[deriv_wx_idx];

    float sum = neuron->b;

    for (int i = 0; i < neuron->W; ++i) sum += neuron->w[i] * x[i];

    return sum;
}

float* update_gradients(Neuron* neuron, float* x, float* cost_act_derivs, int num_derivs) {
    return NULL;
}

float* update_gradients_sm(Neuron* neuron, float* x, float* cost_outputs_derivs, int num_outputs, float* out_z) {
    return NULL;
}