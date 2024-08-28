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

void mlp_test(MLP* mlp, const char* test_name, float inputs[], float correct_outputs[], int num_outputs, FILE* output_file) {
    fprintf(output_file, "%s Test:\n", test_name);
    fprintf(output_file, "------------------------------\n");

    compute_mlp_output(mlp, inputs);

    for (int i = 0; i < num_outputs; ++i) {
        fprintf(output_file, "\t- MLP Output %d: %f\n", i + 1, mlp->outputs[i]);
        fprintf(output_file, "\t- Actual Output %d: %f\n", i + 1, correct_outputs[i]);

        assert(round_to_place(mlp->outputs[i], 5) == round_to_place(correct_outputs[i], 5));
        fprintf(output_file, "\t- Output %d is correct\n", i + 1);
        fprintf(output_file, "------------------------------\n");
    }

    fprintf(output_file, "\n");
}

int main() {
    FILE* output_file = fopen("mlp_tests.txt", "w");

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

    mlp_test(single_output_mlp, "SingleOutputMLPOne", single_output_input_one, single_output_key_one, 1, output_file);
    mlp_test(single_output_mlp, "SingleOutputMLPTwo", single_output_input_two, single_output_key_two, 1, output_file);

    mlp_test(multiple_output_mlp, "MultipleOutputMLPOne", multiple_output_input_one, multiple_output_key_one, 2, output_file);
    mlp_test(multiple_output_mlp, "MultipleOutputMLPOne", multiple_output_input_two, multiple_output_key_two, 2, output_file);

    /* Backward pass tests */
    cost_function backprop_test_cost_func = mse;

    fclose(output_file);
}