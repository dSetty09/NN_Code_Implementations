/**** This function tests the feed forwardfunctionality of a neuron_infinity in a multilayer perceptron with 
 **** respect to infinity
 ****/

#include <iostream>
#include <cassert>
#include <limits>
#include <fstream>

#include "mlp_neuron.hpp"
#include "../activation_functions/C++/nonlinear_functions.hpp"

// Function for rounding to a certain number of decimal places
float round_to_place(float val, float place) {
    float multiple = std::powl(10, place);
    return std::round(val * multiple) / multiple;
}


int main() {
    /*
    std::ofstream output_file("mlp_neuron_infinity_tests.txt");

    std::vector<float> inputs_infinity{-1.1, 12, 14, std::numeric_limits<float>::infinity()};

    std::vector<float> weights_infinity{-0.1, 1.2, 4.5, 2.2};

    float bias_infinity = -10;

    MLPNeuron neuron_infinity{inputs_infinity, weights_infinity, bias_infinity, sigmoid};

    float expected = round_to_place(neuron_infinity.output(), 5);
    float actual = round_to_place(1, 5);

    output_file << "MLP feedforward test with infinity: " << std::endl;
    output_file << "\t- Expected: " << expected << std::endl;
    output_file << "\t- Actual: " << actual << std::endl;
    assert(expected == actual);
    output_file << "PASSED" << "\n\n";

    std::vector<float> inputs{-0.9, 1.2, 0.4, 3.2};

    std::vector<float> weights{-0.1, 1.2, 4.5, 2.2};

    float bias = -10;

    MLPNeuron neuron_normal{inputs, weights, bias, sigmoid};

    expected = round_to_place(neuron_normal.output(), 5);
    actual = round_to_place(0.99997, 5);

    output_file << "MLP feedforward normal test: " << std::endl;
    output_file << "\t- Expected: " << expected << std::endl;
    output_file << "\t- Actual: " << actual << std::endl;
    assert(expected == actual);
    output_file << "PASSED" << "\n\n";

    output_file.close();
    */
}