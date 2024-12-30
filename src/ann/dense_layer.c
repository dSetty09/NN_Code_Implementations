#include "../../include/ann/dense_layer.h"

void forward_pass(DenseLayer* dl, float* x) {
    for (int j = 0; j < dl->num; ++j) 
        (dl->neurons + j)->output = dl->neurons[j].act_func(wsum(dl->neurons + j, x, NO_DERIV, 0, 0, 0), 0);
}

void backprop(DenseLayer* dl, float* x, float** costs_outputs_derivs, int num_next_layer) {
    float** costs_acts_deriv = (float**) malloc(sizeof(float*) * dl->num); 
    for (int j = 0; j < dl->num; ++j) costs_acts_deriv = (float*) malloc(sizeof(float*) * num_next_layer);

    for (int h = 0; h < num_next_layer; ++h) 
        for (int j = 0; j < dl->num; ++j) 
            costs_acts_deriv[j][h] = costs_outputs_derivs[h][j];

    for (int h = 0; h < num_next_layer; ++h) free(costs_outputs_derivs[h]); 
    free(costs_outputs_derivs);

    for (int j = 0; j < dl->num; ++j) {
        float* cost_inputs_derivs = update_gradients(dl->neurons + j, x, costs_acts_deriv[j], num_next_layer);
    }
}