#ifndef ANN_H
#define ANN_H

#include "dense_layer.h"

#ifdef __cplusplus 
extern "C" {
#endif

/* GLOBAL CONSTANTS */

static const unsigned int MSE = 0;
static const unsigned int CROSS_ENTROPY = 1; 


/* DEFINED STRUCTS */

typedef struct ann {
    DenseLayer* layers; // the computing layers (i.e. hidden layers + output layer)
    int num_layers; // the number of computing layers

    float* inputs; // the "input" layer consisting of inputs
    int num_inputs; // the number of inputs

    int batch_size; // size of batch while learning via gradient descent
    int cost_func; // the cost function
} ArtificialNeuralNetwork;


/*
 * Initializes a new ANN according to the specifications declared by the user.
 *
 * @param num_layers | The number of layers
 * @param num_inputs | The number of inputs fed into the ANN
 * @param cost_func | The cost function for the ANN
 * @param neuron_params | A 2d array of neuron parameters, where each row in the array corresponds to a layer
 * @param num_per_layer | An array storing the number of neurons per layer
 * 
 * @return A reference to the initialized ANN
 */

/*
 * Enables an artificial neural network to learn from a given set or subset of training data.
 *
 * @param ann | The artificial neural network being trained
 * @param X_train | The inputs data for training the model
 * @param y_train | The output data for training the model 
 * @param num_data | The total number of training data
 * @param batch_size | The size of the training batches
 * @param ec | The end condition for backpropagation learning
 */
void learn_classifier(ArtificialNeuralNetwork* ann, float** X_train, float* y_train, 
           int num_data, int batch_size, int num_epochs, int ec); 

#ifdef __cplusplus
}
#endif

#endif