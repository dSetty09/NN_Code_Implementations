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

#ifdef __cplusplus
}
#endif

#endif