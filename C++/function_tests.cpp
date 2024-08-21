/* Notes:
 * --> This file tests each of the implemented activation functions
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include "linear_functions.hpp"
#include "nonlinear_functions.hpp"

// Function for rounding to a certain number of decimal places
long double round_to_place(long double val, long double place) {
    long double multiple = std::powl(10, place);
    return std::round(val * multiple) / multiple;
}

// Test function that conducts tests for a specific activation function 
void activation_function_test(long double (*activation_function) (long double), 
                              const char* func_name, long double correct_outputs[]) {

    std::cout << func_name << " Function Tests:\n" << std::endl;

    assert(round_to_place(activation_function(0), 5) == round_to_place(correct_outputs[0], 5));
    std::cout << "--> " << func_name << " function with 0 as argument evaluates to " 
              << correct_outputs[0] << std::endl;
    
    assert(round_to_place(activation_function(-0.4), 5) == round_to_place(correct_outputs[1], 5));
    std::cout << "--> " << func_name << " function with small negative number as arg " 
              << "evaluates to " << correct_outputs[1] << std::endl;

    assert(round_to_place(activation_function(-16), 5) == round_to_place(correct_outputs[2], 5));
    std::cout << "--> " << func_name << " function with large negative number as arg " 
              << "evaluates to " << correct_outputs[2] << std::endl;

    assert(round_to_place(activation_function(0.7), 5) == round_to_place(correct_outputs[3], 5));
    std::cout << "--> " << func_name << " function with small positive number as arg " 
              << "evaluates to " << correct_outputs[3] << std::endl;

    assert(round_to_place(activation_function(22), 5) == round_to_place(correct_outputs[4], 5));
    std::cout << "--> " << func_name << " function with large positive number as arg "
              << "evaluates to " << correct_outputs[4] << '\n' << std::endl;
}


int main() {
    long double linear_test_vals[5] = {0, -0.4, -16, 0.7, 22};
    activation_function_test(linear, "Linear", linear_test_vals); 
     
    long double sigmoid_test_vals[5] = {0.5, 0.40131, 0, 0.66819, 1};
    activation_function_test(sigmoid, "Sigmoid", sigmoid_test_vals); 

    long double tanh_test_vals[5] = {0, -0.37995, -1, 0.60437, 1};
    activation_function_test(hyperbolic_tangent, "Tanh", tanh_test_vals);

    long double step_test_vals[5] = {0, 0, 0, 1, 1};
    activation_function_test(step, "Step", step_test_vals);

    long double relu_test_vals[5] = {0, 0, 0, 0.7, 22};
    activation_function_test(relu, "ReLU", relu_test_vals);

    long double leaky_relu_test_vals[5] = {0, -0.004, -0.16, 0.7, 22};
    activation_function_test(leaky_relu, "Leaky ReLU", leaky_relu_test_vals);

    long double softplus_test_vals[5] = {0.69315, 0.51302, 0, 1.10319, 22};
    activation_function_test(softplus, "SoftPlus", softplus_test_vals);
}
