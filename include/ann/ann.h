#ifndef ANN_H
#define ANN_H

#include "../cost_functions/cost_functions.h"
#include "dense_layer.h"

#ifdef __cplusplus 
extern "C" {
#endif

/* GLOBAL CONSTANTS */

static const unsigned int SE = 0;
static const unsigned int MULTI_CLASS_CROSS_ENTROPY = 1; 


/* DEFINED STRUCTS */

/*
 * An artificial neural network.
 */
typedef struct ann {
    DenseLayer* layers; // the computing layers (i.e. hidden layers + output layer)
    int num_layers; // the number of computing layers

    int num_inputs; // the number of inputs this ANN accepts

    void* cost_function; // the cost function
} ArtificialNeuralNetwork;


/* FUNCTIONS FOR HANDLING ANN STRUCTURE */

/*
 * Initializes a new ANN according to the specifications declared by the user.
 *
 * @param cost_func | The global constant associated with a specific cost function for the ANN.
 * @param num_inputs | The number of inputs this ann accepts.
 * @param num_layers | The number of layers.
 * @params ... | A variable number of dense layers (i.e. arguments of type "DenseLayer").
 * 
 * @warning The behavior of this function is undefined when at least one argument, that is not a "DenseLayer", 
 * is passed as a variable argument
 * 
 * @return An initialized ANN
 */
ArtificialNeuralNetwork* init_ann(int cost_func, int num_inputs, int num_layers, ...);

/*
 * Deletes the memory associated with an artificial neural network.
 *
 * @param ann | An artificial neural network whose associated memory is being deleted.
 */
void del_ann(ArtificialNeuralNetwork* ann);


/* FUNCTIONS FOR ANN OPERATIONS */

/*
 * Returns a set of predictions from this ann for a set of data. 
 *
 * @param ann | The artificial neural network which is making the predictions.
 * @param X | The set of data for which the predictions are being made.
 * @param num_data | The number of data points for which predictions are being made.
 * 
 * @return A set of predictions for the given set of data.
 */
float** predict(ArtificialNeuralNetwork* ann, float** X, int num_data);

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
void fit(ArtificialNeuralNetwork* ann, float** X_train, float* y_train, 
         int num_data, int batch_size, int num_epochs, int ec); 


#ifdef __cplusplus
}
#endif

#endif