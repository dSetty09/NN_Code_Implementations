/**** Note: this file tests forward pass (backpropagation still needs to be done) ****/

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../common_definitions.h"
#include "../activation_functions/C++/linear_functions.h"
#include "../activation_functions/C++/nonlinear_functions.h"
#include "../activation_functions/C++/cost_functions.h"
#include "mlp.h"

#define TRUE 1
#define FALSE 0

void mlp_ff_test(MLP* mlp, const char* test_name, float inputs[], float correct_outputs[], int num_outputs, FILE* output_file) {
    fprintf(output_file, "%s Feed Forward Test:\n", test_name);
    fprintf(output_file, "------------------------------\n");

    compute_mlp_output(mlp, inputs);

    for (int i = 0; i < num_outputs; ++i) {
        fprintf(output_file, "\t- MLP Output %d: %f\n", i + 1, mlp->outputs[i]);
        fprintf(output_file, "\t- Correct Output %d: %f\n", i + 1, correct_outputs[i]);

        assert(round_to_place(mlp->outputs[i], 5) == round_to_place(correct_outputs[i], 5));
        fprintf(output_file, "\t- Output %d is correct\n", i + 1);
        fprintf(output_file, "------------------------------\n");
    }

    fprintf(output_file, "\n");
}

void mlp_bp_test(MLP* mlp, const char* test_name, float inputs[], float ideal_outputs[], cost_function loss_function, int num_outputs, FILE* output_file) {
    fprintf(output_file, "%s Backpropagation Test:\n", test_name);
    fprintf(output_file, "------------------------------\n");

    compute_mlp_output(mlp, inputs);

    fprintf(output_file, "- First Computed Output:\n");
    for (int j = 0; j < num_outputs; ++j) {
        fprintf(output_file, "\t[%d: %f]\n", j, mlp->outputs[j]);
    }

    fprintf(output_file, "- Ideal Output:\n");
    for (int j = 0; j < num_outputs; ++j) {
        fprintf(output_file, "\t[%d: %f]\n", j, ideal_outputs[j]);
    }

    float error = loss_function(mlp->outputs, ideal_outputs, num_outputs, -1);
    fprintf(output_file, "- Error: %f\n", error);

    int i = 1;
    float prev_error = error;

    while (error > 0.005) {
        fprintf(output_file, "------------------------------\n");
        fprintf(output_file, "- Iteration %d:\n", i);

        adjust_weights_and_biases(mlp, loss_function, ideal_outputs);
        compute_mlp_output(mlp, inputs);

        fprintf(output_file, "\t- Computed Outputs after Backpropagation:\n");
        for (int j = 0; j < num_outputs; ++j) {
            fprintf(output_file, "\t[%d: %f]\n", j, mlp->outputs[j]);
        }

        error = loss_function(mlp->outputs, ideal_outputs, num_outputs, -1);
        fprintf(output_file, "\t- Previous Error: %f\n", prev_error);
        fprintf(output_file, "\t- New Error: %f\n", error);

        assert(error < prev_error);
        fprintf(output_file, "\t- Error Reduced Successfully!\n");

        fprintf(output_file, "------------------------------\n");

        prev_error = error;

        ++i;
    } 

    fprintf(output_file, "\n");
}

int main() {
    FILE* output_file = fopen("mlp_tests.txt", "w");

    fprintf(output_file, "FEED FORWARD TESTS:\n\n");

    /* Forward Pass Tests */
    int num_layers = 3;

    float single_output_weights[9] = {0.01, 0.23, -0.47, 0.5, 1.23, -1.79, -0.32, 0.74, -1.11};
    float multiple_output_weights[12] = {0.01, 0.23, -0.47, 0.5, 1.23, -1.79, -0.32, 0.74, -1.11, -0.54, 0.77, 0.14};

    float single_output_input_one[2] = {-0.5, 0.3};
    float single_output_input_two[2] = {-INFINITY, 9};
    int single_output_config[3] = {2, 3, 1}; // neurons per layer for single output MLP 
    float single_output_key_one[1] = {0.53295};
    float single_output_key_two[1] = {1.0};

    float multiple_output_input_one[2] = {0.3, 0.7};
    float multiple_output_input_two[2] = {-INFINITY, 1};
    int multiple_output_config[3] = {2, 3, 2}; // neurons per layer for multiple output MLP
    float multiple_output_key_one[2] = {0.49033, 0.56129};
    float multiple_output_key_two[2] = {1.0, 1.0};

    one_arg_activation_function single_output_function_config[3] = {NULL, softplus, sigmoid};
    one_arg_activation_function multiple_output_function_config[3] = {NULL, softplus, sigmoid};

    MLP* single_output_mlp = build_mlp(num_layers, single_output_config, single_output_function_config, single_output_weights);

    MLP* multiple_output_mlp = build_mlp(num_layers, multiple_output_config, multiple_output_function_config, multiple_output_weights);

    mlp_ff_test(single_output_mlp, "SingleOutputMLPOne", single_output_input_one, single_output_key_one, 1, output_file);
    mlp_ff_test(single_output_mlp, "SingleOutputMLPTwo", single_output_input_two, single_output_key_two, 1, output_file);

    mlp_ff_test(multiple_output_mlp, "MultipleOutputMLPOne", multiple_output_input_one, multiple_output_key_one, 2, output_file);
    mlp_ff_test(multiple_output_mlp, "MultipleOutputMLPOne", multiple_output_input_two, multiple_output_key_two, 2, output_file);

    /* Backward pass tests */
    fprintf(output_file, "\nBACKWARD PROPAGATION TESTS:\n\n");

    cost_function backprop_test_cost_func = mse;

    float single_output_ideal_output_one[1] = {1.0};
    float single_output_ideal_output_two[1] = {0.0};

    float multiple_output_ideal_outputs[2] = {1.0, 0.0};

    mlp_bp_test(single_output_mlp, "SingleOutputMLP", single_output_input_one, single_output_ideal_output_one, backprop_test_cost_func,  1, output_file);
    mlp_bp_test(single_output_mlp, "SingleOutputMLP", single_output_input_one, single_output_ideal_output_two, backprop_test_cost_func,  1, output_file); // testing how model learns with decently large error
    mlp_bp_test(single_output_mlp, "SingleOutputMLP", single_output_input_one, single_output_ideal_output_two, backprop_test_cost_func,  1, output_file); // testing how model learns with very small error

    mlp_bp_test(multiple_output_mlp, "MultipleOutputMLP", multiple_output_input_one, multiple_output_ideal_outputs, backprop_test_cost_func, 2, output_file);

    fclose(output_file);
}