#include "../../include/ann/ann.h"


/* FUNCTIONS FOR HANDLING ANN STRUCTURE */

ArtificialNeuralNetwork* init_ann(int cost_func, int num_layers, ...) {
    ArtificialNeuralNetwork* ret = (ArtificialNeuralNetwork*) malloc(sizeof(ArtificialNeuralNetwork));
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


/** FUNCTIONS FOR NEURAL NETWORK OPERATIONS **/

float** prep_training_data(float X_train_raw[__NUM_TRAIN__][__DATAPOINT_SIZE__]) {
    float** ret = (float**) malloc(sizeof(float*) * __DATAPOINT_SIZE__);

    for (int i = 0; i < __DATAPOINT_SIZE__; ++i) {
        ret[i] = (float*) malloc(sizeof(float) * __DATAPOINT_SIZE__);
        memcpy(ret[i], X_train_raw[i], sizeof(float) * __DATAPOINT_SIZE__);
    }

    return ret;
}

void discard_data(int nrows, float** X) {
    for (int r = 0; r < nrows; ++r) free(X[r]);
    free(X);
}


/* FUNCTIONS FOR ANN FORWARD PASS */

float** alloc_predictions(int num_predictions) {
    float** predictions = (float**) malloc(sizeof(float*) * num_predictions);
    for (int i = 0; i < num_predictions; ++i) predictions[i] = NULL;
    return predictions;
}

void record_predictions(ArtificialNeuralNetwork* ann, float** X, int num_data, float** predictions) {
    for (int i = 0; i < num_data; ++i) {
        int num_inputs_passed = ann->layers[0].neurons->num_weights;
        float* received_data = (float*) malloc(sizeof(float) * num_inputs_passed);
        for (int j = 0; j < num_inputs_passed; ++j) received_data[j] = X[i][j];

        for (int l = 0; l < ann->num_layers; ++l) {
            float* passed_output = NULL;
            int on_output_layer = l == ann->num_layers - 1;

            start_forward_pass(ann->layers + l, &passed_output);

            conduct_forward_pass(ann->layers[l].neurons, ann->layers[l].num_neurons, received_data, 
                                 on_output_layer, passed_output);

            free(received_data);

            received_data = passed_output;
        }

        predictions[i] = received_data;
    }
}

void discard_predictions(float** predictions, int num_predictions) {
    for (int i = 0; i < num_predictions; ++i) free(predictions[i]);
    free(predictions);
}


/* FUNCTIONS SPECIFIC TO CLASSIFICATION */

int* make_classifications(float** predictions, int num_predictions, int num_classes) {
    int* ret = (int*) calloc(num_predictions, sizeof(int));

    for (int i = 0; i < num_predictions; ++i) 
        for (int j = 0; j < num_classes; ++j) 
            if (predictions[j] > predictions[ret[i]]) 
                ret[i] = j;

    return ret;
}

void discard_classifications(int* classifications) {
    free(classifications);
}


/* FUNCTIONS FOR BACKPROPAGATION */

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