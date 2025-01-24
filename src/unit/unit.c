#include "../../include/unit/unit.h"


/* NEURON HANDLING FUNCTIONS */

NeuronNode init_neuron(Neuron spec) {
    NeuronNode ret = {.w=(float*) malloc(sizeof(float) * spec.num_weights), .num_weights=spec.num_weights, 
                      .delta_w=(float*) calloc(spec.num_weights, sizeof(float)),
                      .b=spec.bias, .delta_b=0,
                      .act_func=spec.act_func, .deriv_a=0,
                      .output=0};
    
    for (int i = 0; i < spec.num_weights; ++i) ret.w[i] = (spec.weights) ? spec.weights[i] : random_num(0, 1, 5);

    return ret;
}

void del_neuron(NeuronNode* neuron) {
    free(neuron->w);
    free(neuron->delta_w);
}


/* NEURON OPERATION FUNCTIONS */

float wsum(NeuronNode* neuron, float* x, int deriv_wx_idx, int deriv_w, int deriv_x, int deriv_b) {
    if (deriv_b) return 1;
    if (deriv_w) return x[deriv_wx_idx];
    if (deriv_x) return neuron->w[deriv_wx_idx];

    float sum = neuron->b;

    for (int i = 0; i < neuron->num_weights; ++i) sum += neuron->w[i] * x[i];

    return sum;
}