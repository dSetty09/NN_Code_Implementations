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
        float my_func_output = round_to_place(activation_function(inputs[i], deriv), rounded_place);
        float actual_func_output = round_to_place(correct_outputs[i], rounded_place);

        fprintf(output_file, "\t> Test %d:\n", i);
        fprintf(output_file, "\t\t- Mine: %s(%f) ==> %f\n", func_name, inputs[i], my_func_output);
        fprintf(output_file, "\t\t- Actual: %s(%f) ==> %f\n", func_name, inputs[i], actual_func_output);

        if (isnan(my_func_output)) assert(isnan(actual_func_output));
        else assert(my_func_output == actual_func_output);

        fprintf(output_file, "\t\t- %s function with %f as argument evaluates to %f\n\n",
                func_name, inputs[i], actual_func_output);
    }
}


int main() {
    FILE* output_file = fopen("function_tests.txt", "w");

    float input_vals[7] = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY};

    int num_tests = 7;

    float linear_test_vals[7] = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY};
    float linear_deriv_vals[7] = {1, 1, 1, 1, 1, 1, 1};
    activation_function_test(linear, "Linear", input_vals, linear_test_vals, num_tests, output_file, FALSE);
    activation_function_test(linear, "LinearDerivative", input_vals, linear_deriv_vals, num_tests, output_file, TRUE); 
     
    float sigmoid_test_vals[7] = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};
    float sigmoid_deriv_vals[7] = {0.25, 0.24026, 0, 0.22171, 0, NAN, 0};
    activation_function_test(sigmoid, "Sigmoid", input_vals, sigmoid_test_vals, num_tests, output_file, FALSE); 
    activation_function_test(sigmoid, "SigmoidDerivative", input_vals, sigmoid_deriv_vals, num_tests, output_file, TRUE);

    float tanh_test_vals[7] = {0, -0.37995, -1, 0.60437, 1, -1, 1};
    float tanh_deriv_vals[7] = {1, 0.85564, 0, 0.63474, 0, 0, 0};
    activation_function_test(hyperbolic_tangent, "Tanh", input_vals, tanh_test_vals, num_tests, output_file, FALSE);
    activation_function_test(hyperbolic_tangent, "TanhDerivative", input_vals, tanh_deriv_vals, num_tests, output_file, TRUE);

    float step_test_vals[7] = {0, 0, 0, 1, 1, 0, 1};
    float step_deriv_vals[7] = {NAN, 0, 0, 0, 0, 0, 0};
    activation_function_test(step, "Step", input_vals, step_test_vals, num_tests, output_file, FALSE);
    activation_function_test(step, "StepDerivative", input_vals, step_deriv_vals, num_tests, output_file, TRUE);

    float relu_test_vals[7] = {0, 0, 0, 0.7, 22, 0, INFINITY};
    float relu_deriv_vals[7] = {NAN, 0, 0, 1, 1, 0, 1};
    activation_function_test(relu, "ReLU", input_vals, relu_test_vals, num_tests, output_file, FALSE);
    activation_function_test(relu, "ReLUDerivative", input_vals, relu_deriv_vals, num_tests, output_file, TRUE);

    float leaky_relu_test_vals[7] = {0, -0.004, -0.16, 0.7, 22, -INFINITY, INFINITY};
    float leaky_relu_deriv_vals[7] = {NAN, 0.01, 0.01, 1, 1, 0.01, 1};
    activation_function_test(leaky_relu, "LeakyReLU", input_vals, leaky_relu_test_vals, num_tests, output_file, FALSE);
    activation_function_test(leaky_relu, "LeakyReLUDerivative", input_vals, leaky_relu_deriv_vals, num_tests, output_file, TRUE);

    float softplus_test_vals[7] = {0.69315, 0.51302, 0, 1.10319, 22, 0, INFINITY};
    float softplus_deriv_vals[7] = {0.5, 0.40131, 0, 0.66819, 1, 0, NAN};
    activation_function_test(softplus, "SoftPlus", input_vals, softplus_test_vals, num_tests, output_file, FALSE);
    activation_function_test(softplus, "SoftPlusDerivative", input_vals, softplus_deriv_vals, num_tests, output_file, TRUE);

    fclose(output_file);
}
