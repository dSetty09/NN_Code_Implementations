#ifndef ANN_H
#define ANN_H

#include "../cost_functions/cost_functions.h"
#include "dense_layer.h"

#ifdef __cplusplus 
extern "C" {
#endif


/* GLOBAL CONSTANTS  */
static const unsigned int SE = 0;
static const unsigned int MULTI_CLASS_CROSS_ENTROPY = 1; 

static const unsigned int PAST_MAX_EPOCHS = 0;
static const unsigned int LESS_THAN_MIN_DIFF = 1;
static const unsigned int MEETS_FAIR_ERROR = 2;


/* FUNCTIONS FOR TESTING END CONDITIONS TO BACKPROPAGATION */

/*
 * Tests whether the end condition value is past a specified max allowed number of epochs. 
 *
 * @param epochs_passed | The current number of epochs passed.
 * @param max_epochs | The max allowed number of epochs.
 * 
 * @return True if the max allowed number of epochs has been passed or false otherwise.
 */
int past_max_epochs(float epochs_passed, float max_epochs);

/*
 * Tests whether the error of the neural network is changing significantly (i.e. if the current difference
 * between the last and current error is larger than the minimum difference)
 * 
 * @param curr_diff | Current difference between the last and current error. 
 * @param min_diff | Minimum difference that must be met.
 * 
 * @return True if the minimum difference has been met or false otherwise.
 */
int less_than_min_diff(float curr_diff, float min_diff);

/*
 * Tests whether the current error of the neural network is less than or equal to a specific fair error.
 *
 * @param curr_error | Current error. 
 * @param fair_error | Fair error.
 * 
 * @return True if the fair error has been met or false otherwise.
 */
int meets_fair_error(float curr_error, float fair_error);

typedef int (*end_condition) (float, float); // type definition corresponding to a pointer to an "end condition"


/* DEFINED STRUCTS */

/*
 * An artificial neural network.
 */
typedef struct ann {
    DenseLayer* layers; // the computing layers (i.e. hidden layers + output layer)
    int num_layers; // the number of computing layers

    int cost_func; // cost function 
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
 * @param ann | The artificial neural network whose associated memory is being deleted.
 */
void del_ann(ArtificialNeuralNetwork* ann);


/* FUNCTIONS FOR ANN VISUALIZATION */

/*
 * Displays all the parameters for a given artificial neural network in a neat format.
 *
 * @param ann | The artificial neural network whose parameters are being displayed.
 */
void display_parameters(ArtificialNeuralNetwork* ann);


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

/*
 * One-hot encodes a given array of classifications.
 *
 * @param y | The array which is being one-hot encoded.
 * @param size | The size of the array.
 * 
 * @return The one-hot encoded matrix representing the given array
 */
float** one_hot_encoded_mat(float* y, int size);

/*
 * Frees the associated memory with the one hot encoded matrix.
 *
 * @param y_one_hot | The matrix whose associated memory is being freed.
 * @param size | The size of the matrix in terms of number of rows.
 */
void discard_one_hot_encoded_mat(float** y_one_hot, int size);


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
void record_predictions(ArtificialNeuralNetwork* ann, float** X, int num_data, float** predictions);

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
 * Resets the gradients of the given artificial neural network back to 0.
 *
 * @param ann | The artificial neural network whose gradients are being modified.
 */
void reset_gradients(ArtificialNeuralNetwork* ann);

/*
 * Conduct gradient descent on a batch of training data.
 *
 * @param ann | The artificial neural network whose gradients are being updated.
 * @param X_train | The inputs data associated with a specific batch from the training data.
 * @param y_train_enc | The output data associated with a specific batch from the training data, one-hot encoded.
 */
void calc_gradients(ArtificialNeuralNetwork* ann, float** X_train, float** y_train_enc);

/*
 * Updates the weight and bias parameters throughout the network according to the updated gradients.
 *
 * @param ann | The ann whose weights and biases are being updated.
 */
void adjust_weights_and_biases(ArtificialNeuralNetwork* ann);

/*
 * Updates parameters of artificial neural network to learn from a given set or subset of training data.
 *
 * @param ann | The artificial neural network being trained
 * @param X_train | The inputs data for training the model
 * @param y_train | The output data for training the model 
 * @param batch_size | The size of the training batches
 * @param ec | The end condition for backpropagation learning
 * @param ec_criteria | The end condition criteria
 */
void fit(ArtificialNeuralNetwork* ann, float** X_train, float* y_train, int batch_size, int ec, float ec_criteria); 


#ifdef __cplusplus
}
#endif

#endif