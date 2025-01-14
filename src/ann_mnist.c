/* FILE FOR TRAINING ANN ON MNIST */

#include "../include/mnist.h"

#include "../include/neural_net_ops/neural_net_ops.h"

#include "../include/activation_functions/nonlinear_functions.h"

#include "../include/cost_functions/cost_functions.h"

#include "../include/unit/unit.h"

#include "../include/ann/dense_layer.h"
#include "../include/ann/ann.h"

// int main() {
    // load_mnist();

    // ArtificialNeuralNetwork* ann = create_ann(3, 2, CROSS_ENTROPY);
    
    // TODO: create each neuron in each layer
    // layers->neurons = (Neuron*) calloc(2, sizeof(Neuron));
    // for (int i = 0; i < 2; ++i) {
    //     Neuron* curr_neuron = layers->neurons;
    //     layers->neurons 
    // }

    // layers->num = 2;

    // (layers + 1)->neurons = (Neuron*) calloc(3, sizeof(Neuron));
    // (layers + 1)->num = 3;

    // (layers + 2)->neurons = (Neuron*) calloc(2, sizeof(Neuron));
    // (layers + 2)->num = 2;

    // float** X_train = NULL;
    // float* y_train = NULL;

    // learn_classifier(ann, X_train, y_train, NUM_TRAIN, 250, 50, PAST_MAX_EPOCHS);

    // return 0;
// }