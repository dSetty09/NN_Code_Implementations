/**** Note: this file tests each of the implemented activation functions ****/

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../../common_definitions.h"
#include "linear_functions.h"
#include "nonlinear_functions.h"

#define TRUE 1
#define FALSE 0

// Test function that conducts tests for a specific activation function 
void activation_function_test(one_arg_activation_function activation_function, const char* func_name, 
                              float inputs[], float correct_outputs[], int num_tests,
                              FILE* output_file, int deriv) {

    float rounded_place = 5;

    fprintf(output_file, "%s Function Tests:\n", func_name);

    for (int i = 0; i < num_tests; ++i) {
        float my_func_output = activation_function(inputs[i], deriv);
        float actual_func_output = correct_outputs[i];

        fprintf(output_file, "\t> Test %d:\n", i);
        fprintf(output_file, "\t\t- Mine: %s(%f) ==> %f\n", func_name, inputs[i], my_func_output);
        fprintf(output_file, "\t\t- Actual: %s(%f) ==> %f\n", func_name, inputs[i], actual_func_output);

        if (my_func_output == NEAR_ZERO) assert(my_func_output == actual_func_output);
        else assert(are_similar(my_func_output, actual_func_output));

        fprintf(output_file, "\t\t- %s function with %f as argument evaluates to %f\n\n",
                func_name, inputs[i], actual_func_output);
    }
}


int main() {
    FILE* output_file = fopen("function_tests.txt", "w");

    float input_vals[7] = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY};

    int num_tests = 7;

    float linear_test_vals[7] = {0, -0.4, -16, 0.7, 22, -FLT_MAX, FLT_MAX};
    float linear_deriv_vals[7] = {1, 1, 1, 1, 1, 1, 1};
    activation_function_test(linear, "Linear", input_vals, linear_test_vals, num_tests, output_file, FALSE);
    activation_function_test(linear, "LinearDerivative", input_vals, linear_deriv_vals, num_tests, output_file, TRUE); 
     
    float sigmoid_test_vals[7] = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};
    float sigmoid_deriv_vals[7] = {0.25, 0.24026, NEAR_ZERO, 0.22171, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO};
    activation_function_test(sigmoid, "Sigmoid", input_vals, sigmoid_test_vals, num_tests, output_file, FALSE); 
    activation_function_test(sigmoid, "SigmoidDerivative", input_vals, sigmoid_deriv_vals, num_tests, output_file, TRUE);

    float tanh_test_vals[7] = {0, -0.37995, -1, 0.60437, 1, -1, 1};
    float tanh_deriv_vals[7] = {1, 0.85564, NEAR_ZERO, 0.63474, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO};
    activation_function_test(hyperbolic_tangent, "Tanh", input_vals, tanh_test_vals, num_tests, output_file, FALSE);
    activation_function_test(hyperbolic_tangent, "TanhDerivative", input_vals, tanh_deriv_vals, num_tests, output_file, TRUE);

    float step_test_vals[7] = {0, 0, 0, 1, 1, 0, 1};
    float step_deriv_vals[7] = {NEAR_ZERO, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO, NEAR_ZERO};
    activation_function_test(step, "Step", input_vals, step_test_vals, num_tests, output_file, FALSE);
    activation_function_test(step, "StepDerivative", input_vals, step_deriv_vals, num_tests, output_file, TRUE);

    float relu_test_vals[7] = {0, 0, 0, 0.7, 22, 0, FLT_MAX};
    float relu_deriv_vals[7] = {NEAR_ZERO, NEAR_ZERO, NEAR_ZERO, 1, 1, NEAR_ZERO, 1};
    activation_function_test(relu, "ReLU", input_vals, relu_test_vals, num_tests, output_file, FALSE);
    activation_function_test(relu, "ReLUDerivative", input_vals, relu_deriv_vals, num_tests, output_file, TRUE);

    float leaky_relu_test_vals[7] = {0, -0.004, -0.16, 0.7, 22, -FLT_MAX * 0.01, FLT_MAX};
    float leaky_relu_deriv_vals[7] = {0.01, 0.01, 0.01, 1, 1, 0.01, 1};
    activation_function_test(leaky_relu, "LeakyReLU", input_vals, leaky_relu_test_vals, num_tests, output_file, FALSE);
    activation_function_test(leaky_relu, "LeakyReLUDerivative", input_vals, leaky_relu_deriv_vals, num_tests, output_file, TRUE);

    float softplus_test_vals[7] = {0.69315, 0.51302, 0, 1.10319, 22, 0, FLT_MAX};
    float softplus_deriv_vals[7] = {0.5, 0.40131, NEAR_ZERO, 0.66819, 1, NEAR_ZERO, 1};
    activation_function_test(softplus, "SoftPlus", input_vals, softplus_test_vals, num_tests, output_file, FALSE);
    activation_function_test(softplus, "SoftPlusDerivative", input_vals, softplus_deriv_vals, num_tests, output_file, TRUE);

    ReadVectFmt* expected_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));
    ReadVectFmt* actual_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));

    float softmax_test_input[5] = {12, -7, 3, 4, 8};

    float softmax_test_expected[5] = {0.981571, 0, 0.000121, 0.000329, 0.017978};
    expected_rvf->vect = softmax_test_expected;
    expected_rvf->num_rows = 1;
    expected_rvf->num_cols = 5;
    expected_rvf->num_layers = 1;

    float softmax_test_actual[5] = {softmax(softmax_test_input, 0, 5, NO_DERIV), softmax(softmax_test_input, 1, 5, NO_DERIV),
                                    softmax(softmax_test_input, 2, 5, NO_DERIV), softmax(softmax_test_input, 3, 5, NO_DERIV),
                                    softmax(softmax_test_input, 4, 5, NO_DERIV)};
    actual_rvf->vect = softmax_test_actual;
    actual_rvf->num_rows = 1;
    actual_rvf->num_cols = 5;
    actual_rvf->num_layers = 1;
    
    disp_test_results("SOFTMAX TEST", "FUNCTION OUTPUT", (void*) expected_rvf, (void*) actual_rvf, FALSE, output_file);

    float softmax_deriv_expected[5] = {0.0180889, 0.017654896, 0.00012112, -0.0000021777879, -0.017646798};
    expected_rvf->vect = softmax_deriv_expected;
    expected_rvf->num_rows = 1;
    expected_rvf->num_cols = 5;
    expected_rvf->num_layers = 1;

    float softmax_deriv_actual[5] = {softmax(softmax_test_input, 0, 5, 0), softmax(softmax_test_input, 4, 5, 4),
                                     softmax(softmax_test_input, 2, 5, 2), softmax(softmax_test_input, 2, 5, 4),
                                     softmax(softmax_test_input, 4, 5, 0)};
    actual_rvf->vect = softmax_deriv_actual;
    actual_rvf->num_rows = 1;
    actual_rvf->num_cols = 5;
    actual_rvf->num_layers = 1;

    disp_test_results("SOFTMAX TEST", "FUNCTION DERIVATIVE", (void*) expected_rvf, (void*) actual_rvf, FALSE, output_file);

    free(expected_rvf);
    free(actual_rvf);

    fclose(output_file);
}
