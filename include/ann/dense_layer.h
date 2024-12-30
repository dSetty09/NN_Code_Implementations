#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H 

#include "../unit/unit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl {
    Neuron* neurons;
    int num;
} DenseLayer;

/*
 * Conducts a forward pass operation for each neuron in the dense layer. 
 *
 * @param dl | The dense layer of concern.
 * @param x | The array of inputs given to this layer. 
 */
void forward_pass(DenseLayer* dl, float* x);

/*
 * DESCRIPTION DUE SOON
 */
void backprop(DenseLayer* dl, float** costs_inputs_derivs, int num_cost_act_derivs);

#ifdef __cplusplus
}
#endif

#endif