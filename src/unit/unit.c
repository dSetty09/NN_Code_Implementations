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
    neuron->delta_a = 0;
    for (int i = 0; i < num_derivs; ++i) neuron->delta_a += cost_act_derivs[i];  

    float cost_wsum_deriv = neuron->delta_a * neuron->act_func(wsum(neuron, x, NO_DERIV, 0, 0, 0), 1);

    for (int j = 0; j < neuron->W; ++j) 
        neuron->delta_w[j] = cost_wsum_deriv * wsum(neuron, x, j, 1, 0, 0); 

    neuron->delta_b = cost_wsum_deriv * wsum(neuron, x, NO_DERIV, 0, 0, 1);

    float* cost_inputs_derivs = (float*) malloc(sizeof(float) * neuron->W);
    for (int k = 0; k < neuron->W; ++k) cost_inputs_derivs[k] = cost_wsum_deriv * wsum(neuron, x, k, 0, 1, 0); 

    return cost_inputs_derivs;
}