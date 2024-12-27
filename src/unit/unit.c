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

void update_gradients(Neuron* neuron, float* x, float cost_to_wsum_deriv) {
    neuron->delta_b = cost_to_wsum_deriv * wsum(neuron, x, NO_DERIV, 0, 0, 1);

    for (int i = 0; i < neuron->W; ++i) neuron->delta_w[i] = cost_to_wsum_deriv * wsum(neuron, x, i, 1, 0, 0); 
}