#include "../../include/ann/ann.h"


/* FUNCTIONS FOR HANDLING ANN STRUCTURE */

ArtificialNeuralNetwork* init_ann(int cost_func, int num_inputs, int num_layers, ...) {
    ArtificialNeuralNetwork* ret = (ArtificialNeuralNetwork*) malloc(sizeof(ArtificialNeuralNetwork));
    ret->num_inputs = num_inputs;
    ret->layers = NULL; ret->layers = (DenseLayer*) malloc(sizeof(DenseLayer) * num_layers);
    ret->num_layers = num_layers;

    switch (cost_func) {
        case SE:
            ret->cost_function = (void*) se; 
            break;
        default:
            ret->cost_function = (void*) multiclass_ce; 
    }

    va_list layers;
    va_start(layers, num_layers);

    for (int i = 0; i < num_layers; ++i) ret->layers[i] = va_arg(layers, DenseLayer); 

    va_end(layers);

    return ret;
}

void del_ann(ArtificialNeuralNetwork* ann) {
    for (int i = 0; i < ann->num_layers; ++i) del_dense_layer(ann->layers + i);
    free(ann->layers);
}


/* FUNCTIONS FOR NEURAL NETWORK OPERATIONS */

void learn_classifier(ArtificialNeuralNetwork* ann, float** X_train, float* y_train, 
           int num_data, int batch_size, int num_epochs, int ec) {

    int num_batches = 0;
    int* num_per_batch = NULL;
    int** batches = generate_training_batches(num_data, batch_size, &num_batches, &num_per_batch);

    int end = FALSE; // flag indicating whether should end learning or not

    for (int b = 0; b < num_batches && !end; ++b) {
        for (int d = 0; d < num_per_batch[b]; ++d) {
            int data_index = batches[b][d];
            
            float* input = X_train[data_index];
            float actual_output = y_train[data_index];
        }

        switch (ec) {
        case PAST_FAIR_ERROR:
            if (past_fair_error(0, 0)) end = TRUE; 
            break;
        case PAST_MIN_DIFF:
            if (past_min_diff(0, 0, 0)) end = TRUE;
            break;
        default:
            if (past_max_epochs(0, 0)) end = TRUE;
            break;
        }
    } 
}