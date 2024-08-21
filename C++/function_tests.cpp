/**** Note: this file tests each of the implemented activation functions ****/

#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <fstream>

#include "linear_functions.hpp"
#include "nonlinear_functions.hpp"

// Function for rounding to a certain number of decimal places
float round_to_place(float val, float place) {
    float multiple = std::powl(10, place);
    return std::round(val * multiple) / multiple;
}

// Test function that conducts tests for a specific activation function 
void activation_function_test(float (*activation_function) (float), const char* func_name, 
                              float inputs[], float correct_outputs[], int num_tests,
                              std::ofstream& output_file) {

    float rounded_place = 5;

    output_file << func_name << " Function Tests:" << std::endl;

    for (int i = 0; i < num_tests; ++i) {
        float my_func_output = round_to_place(activation_function(inputs[i]), rounded_place);
        float actual_func_output = round_to_place(correct_outputs[i], rounded_place);

        output_file << "\t> Test " << i << ':' << std::endl;

        output_file << "\t\t- Mine: " << func_name << '(' << inputs[i] << ") ==> " 
                  << my_func_output << std::endl;

        output_file << "\t\t- Actual: " << func_name << '(' << inputs[i] << ") ==> "
                  << actual_func_output << std::endl;

        assert(my_func_output == actual_func_output);
        output_file << "\t\t- " << func_name << " function with " << inputs[i] 
                  << " as argument evaluates to " << correct_outputs[i] << std::endl;
    }
}


int main() {
    std::ofstream output_file("function_tests.txt");

    float input_vals[7] = {0, -0.4, -16, 0.7, 22, -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};

    int num_tests = 7;

    float linear_test_vals[7] = {0, -0.4, -16, 0.7, 22, 
                                       -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::infinity()};
    activation_function_test(linear, "Linear", input_vals, linear_test_vals, num_tests, output_file); 
     
    float sigmoid_test_vals[7] = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};
    activation_function_test(sigmoid, "Sigmoid", input_vals, sigmoid_test_vals, num_tests, output_file); 

    float tanh_test_vals[7] = {0, -0.37995, -1, 0.60437, 1, -1, 1};
    activation_function_test(hyperbolic_tangent, "Tanh", input_vals, tanh_test_vals, num_tests, output_file);

    float step_test_vals[7] = {0, 0, 0, 1, 1, 0, 1};
    activation_function_test(step, "Step", input_vals, step_test_vals, num_tests, output_file);

    float relu_test_vals[7] = {0, 0, 0, 0.7, 22, 0, std::numeric_limits<float>::infinity()};
    activation_function_test(relu, "ReLU", input_vals, relu_test_vals, num_tests, output_file);

    float leaky_relu_test_vals[7] = {0, -0.004, -0.16, 0.7, 22,
                                           -std::numeric_limits<float>::infinity(),
                                           std::numeric_limits<float>::infinity()};
    activation_function_test(leaky_relu, "LeakyReLU", input_vals, leaky_relu_test_vals, num_tests, output_file);

    float softplus_test_vals[7] = {0.69315, 0.51302, 0, 1.10319, 22, 0, 
                                         std::numeric_limits<float>::infinity()};
    activation_function_test(softplus, "SoftPlus", input_vals, softplus_test_vals, num_tests, output_file);

    output_file.close();
}
