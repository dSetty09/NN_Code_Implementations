#include "../../include/ann/ann.h"

/* FUNCTIONS FOR TESTING END CONDITIONS TO BACKPROPAGATION */

int past_max_epochs(float epochs_passed, float max_epochs) {
    return epochs_passed > max_epochs;
}

int less_than_min_diff(float curr_diff, float min_diff) {
    return curr_diff <= min_diff;
}

int meets_fair_error(float curr_error, float fair_error) {
    return curr_error <= fair_error;
}


/* FUNCTIONS FOR HANDLING ANN STRUCTURE */

ArtificialNeuralNetwork* init_ann(int cost_func, int num_layers, ...) {
    ArtificialNeuralNetwork* ret = (ArtificialNeuralNetwork*) malloc(sizeof(ArtificialNeuralNetwork));
    ret->layers = NULL; ret->layers = (DenseLayer*) malloc(sizeof(DenseLayer) * num_layers);
    ret->num_layers = num_layers;
    ret->cost_func = cost_func; 

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


/* FUNCTIONS FOR ANN VISUALIZATION */

void display_parameters(ArtificialNeuralNetwork* ann) {
    printf("ANN Parameters:\n\n");

    for (int l = 0; l < ann->num_layers; ++l) {
        DenseLayer* curr_layer = ann->layers + l;
        printf("Layer %d:\n", l + 1);

        for (int i = 0; i < curr_layer->num_neurons; ++i) {
            NeuronNode* curr_neuron = curr_layer->neurons + i;
            printf("\t* Neuron %d:\n", i + 1);

            printf("\t\t- b=%f\n", curr_neuron->b);
            for (int j = 0; j < curr_neuron->num_weights; ++j) printf("\t\t- w%d=%f\n", j, curr_neuron->w[j]);

            printf("\t\t> delta_b=%f\n", curr_neuron->delta_b); 
            for (int j = 0; j < curr_neuron->num_weights; ++j) 
                printf("\t\t> delta_w%d=%f\n", j, curr_neuron->delta_w[j]);
        }

        printf("\n");
    }
}


/* FUNCTIONS FOR NEURAL NETWORK OPERATIONS */

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

float** one_hot_encoded_mat(float* y, int size) {
    float** ret = (float**) malloc(sizeof(float*) * size);

    for (int i = 0; i < size; ++i) {
        ret[i] = (float*) malloc(sizeof(float) * __NUM_CLASSES__);
        for (int j = 0; j < __NUM_CLASSES__; ++j) ret[i][j] = y[i] == __CLASS_LABELS__[j];
    }

    return ret;
}

void discard_one_hot_encoded_mat(float** y_one_hot, int size) {
    for (int i = 0; i < size; ++i) free(y_one_hot[i]);
    free(y_one_hot);
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

void reset_gradients(ArtificialNeuralNetwork* ann) {
    for (int l = 0; l < ann->num_layers; ++l) {
        for (int i = 0; i < ann->layers[l].num_neurons; ++i) {
            ((ann->layers + l)->neurons + i)->delta_b = 0;
            ((ann->layers + l)->neurons + i)->deriv_a = 0;

            for (int j = 0; j < ann->layers[l].neurons[i].num_weights; ++j)
                ((ann->layers + l)->neurons + i)->delta_w[j] = 0;
        }
    }
}

void calc_gradients(ArtificialNeuralNetwork* ann, float** X_train, float** y_train_enc) {

}

void adjust_weights_and_biases(ArtificialNeuralNetwork* ann) {

}

void fit(ArtificialNeuralNetwork* ann, float** X_train, float* y_train, int batch_size, int ec, float ec_criteria) {
    int ec_met = 0; // flag indicating whether end condition met or not

    int epochs_passed = 0;

    float last_err = INFINITY;
    float curr_err = INFINITY; 

    while (!ec_met) {
        int num_batches = 0;
        int* num_per_batch = NULL;
        int** batches = generate_training_batches(__NUM_TRAIN__, batch_size, &num_batches, &num_per_batch);

        for (int b = 0; b < num_batches; ++b) {
            float** X_train_batch = flt_addr_arr_extract(X_train, batches[b], num_per_batch[b]);

            float* y_train_batch = flt_arr_extract(y_train, batches[b], num_per_batch[b]);
            float** y_train_batch_enc = one_hot_encoded_mat(y_train_batch, num_per_batch[b]);

            reset_gradients(ann);
            calc_gradients(ann, X_train_batch, y_train_batch_enc);
            adjust_weights_and_biases(ann);

            float** y_hat_pdistros = NULL;

            switch (ann->cost_func) {
                case MULTI_CLASS_CROSS_ENTROPY:
                    y_hat_pdistros = alloc_predictions(num_per_batch[b]);

                    record_predictions(ann, X_train_batch, num_per_batch[b], y_hat_pdistros);
                    curr_err = mean_multiclass_ce(y_train_batch_enc, y_hat_pdistros, num_per_batch[b], __NUM_CLASSES__);
                    discard_predictions(y_hat_pdistros, num_per_batch[b]);

                    break;
                default:
                    // do nothing 
            }

            discard_one_hot_encoded_mat(y_train_batch_enc, num_per_batch[b]);

            free(X_train_batch);
            free(y_train_batch);

            ++epochs_passed;

            switch (ec) {
                case PAST_MAX_EPOCHS:
                    ec_met = past_max_epochs(epochs_passed, ec_criteria);
                    break;
                case LESS_THAN_MIN_DIFF:
                    ec_met = less_than_min_diff(fabsf(curr_err - last_err), ec_criteria);
                    break;
                default:
                    ec_met = meets_fair_error(curr_err, ec_criteria);
            }

            last_err = curr_err;
        }

        ++epochs_passed;
    }
}