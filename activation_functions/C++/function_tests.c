/**** Note: this file tests each of the implemented activation functions ****/

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "linear_functions.h"
#include "nonlinear_functions.h"

// Function for rounding to a certain number of decimal places
float round_to_place(float val, float place) {
    float multiple = powl(10, place);
    return round(val * multiple) / multiple;
}

// Test function that conducts tests for a specific activation function 
void activation_function_test(float (*activation_function) (float), const char* func_name, 
                              float inputs[], float correct_outputs[], int num_tests,
                              FILE* output_file) {

    float rounded_place = 5;

    fprintf(output_file, "%s Function Tests:\n", func_name);

    for (int i = 0; i < num_tests; ++i) {
        float my_func_output = round_to_place(activation_function(inputs[i]), rounded_place);
        float actual_func_output = round_to_place(correct_outputs[i], rounded_place);

        fprintf(output_file, "\t> Test %d:\n", i);
        fprintf(output_file, "\t\t- Mine: %s(%f) ==> %f\n", 
                func_name, inputs[i], activation_function(inputs[i]));
        fprintf(output_file, "\t\t- Actual: %s(%f) ==> %f\n",
                func_name, inputs[i], correct_outputs[i]);

        assert(my_func_output == actual_func_output);
        fprintf(output_file, "\t\t- %s function with %f as argument evaluates to %f\n",
                func_name, inputs[i], correct_outputs[i]);
    }
}


int main() {
    FILE* output_file = fopen("function_tests.txt", "w");

    float input_vals[7] = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY};

    int num_tests = 7;

    float linear_test_vals[7] = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY};
    activation_function_test(linear, "Linear", input_vals, linear_test_vals, num_tests, output_file); 
     
    float sigmoid_test_vals[7] = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};
    activation_function_test(sigmoid, "Sigmoid", input_vals, sigmoid_test_vals, num_tests, output_file); 

    float tanh_test_vals[7] = {0, -0.37995, -1, 0.60437, 1, -1, 1};
    activation_function_test(hyperbolic_tangent, "Tanh", input_vals, tanh_test_vals, num_tests, output_file);

    float step_test_vals[7] = {0, 0, 0, 1, 1, 0, 1};
    activation_function_test(step, "Step", input_vals, step_test_vals, num_tests, output_file);

    float relu_test_vals[7] = {0, 0, 0, 0.7, 22, 0, INFINITY};
    activation_function_test(relu, "ReLU", input_vals, relu_test_vals, num_tests, output_file);

    float leaky_relu_test_vals[7] = {0, -0.004, -0.16, 0.7, 22, -INFINITY, INFINITY};
    activation_function_test(leaky_relu, "LeakyReLU", input_vals, leaky_relu_test_vals, num_tests, output_file);

    float softplus_test_vals[7] = {0.69315, 0.51302, 0, 1.10319, 22, 0, INFINITY};
    activation_function_test(softplus, "SoftPlus", input_vals, softplus_test_vals, num_tests, output_file);

    fclose(output_file);
}
