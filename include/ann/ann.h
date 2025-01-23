#ifndef ANN_H
#define ANN_H

#include "../cost_functions/cost_functions.h"
#include "dense_layer.h"

#ifdef __cplusplus 
extern "C" {
#endif


/* GLOBAL CONSTANTS (MODIFY AS YOU WISH) */

static const unsigned int SE = 0;
static const unsigned int MULTI_CLASS_CROSS_ENTROPY = 1; 


/* DEFINED STRUCTS */

/*
 * An artificial neural network.
 */
typedef struct ann {
    DenseLayer* layers; // the computing layers (i.e. hidden layers + output layer)
    int num_layers; // the number of computing layers

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
 * @attention The initialized ANN will select a number of inputs equal to the number of associated weights for 
 * each neuron in the first dense layer, where each neuron has the same number of associated weights.
 * 
 * @return An initialized ANN
 */
ArtificialNeuralNetwork* init_ann(int cost_func, int num_layers, ...);

/*
 * Deletes the memory associated with an artificial neural network.
 *
 * @param ann | An artificial neural network whose associated memory is being deleted.
 */
void del_ann(ArtificialNeuralNetwork* ann);


/** FUNCTIONS FOR ANN OPERATIONS **/

/*
 * Prepares a given set of training_data for being processed by an artificial neural network.
 *
 * @param X_train_raw | An array of training data points.
 * 
 * @return Returns a given set of data as a pointer to pointers, which is a form of data an ANN can process.
 */
float** prep_training_data(float x_train_raw[__NUM_TRAIN__][__DATAPOINT_SIZE__]);

/*
 * Frees the memory that was associated with a given set of data when it was previously being prepped for
 * ANN operations.
 * 
 * @param nrows | The number of rows in said data.
 * @param X | The data for which associated memory is being freed.
 */
void discard_data(int nrows, float** X);


/* FUNCTIONS FOR ANN FORWARD PASS */

/*
 * Returns a buffer needed to store the results for a certain number of predictions.
 *
 * @param num_predictions | The number of predictions being made.
 * 
 * @return A buffer with the allocated memory necessary to store results for a certain number of predictions.
 */
float** alloc_predictions(int num_predictions);

/*
 * Records a set of predictions from this ann for a set of data. 
 *
 * @param ann | The artificial neural network which is making the predictions.
 * @param X | The set of data for which the predictions are being made.
 * @param num_data | The number of data points for which predictions are being made.
 * @param predictions | An array storing the predictions that will be made.
 */
void record_predictions(ArtificialNeuralNetwork* ann, float* X[], int num_data, float** predictions);

/*
 * Frees the memory allocated to store a set of predictions.
 *
 * @param predictions | An array storing the predictions for which memory will be freed. 
 * @param num_predictions | The number of predictions made.
 */
void discard_predictions(float** predictions, int num_predictions);


/* FUNCTIONS SPECIFIC TO CLASSIFICATION */

/*
 * Return an array of classifications that are the most likely for their respective set of predictions.
 *
 * @param predictions | An array storing the likelihoods for each class being predicted. 
 * @param num_predictions | The number of predictions that were made.
 * @param num_classes | The number of classes. 
 * 
 * @return An array of classifications that are the most likely for their respective set of predictions.
 */
int* make_classifications(float** predictions, int num_predictions, int num_classes);

/*
 * Frees the allocated memory associated with an array of classifications.
 *
 * @param classifications | The array of classifications whose associated memory is being freed.
 */
void discard_classifications(int* classifications);


/* FUNCTIONS FOR ANN BACKPROPAGATION */

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