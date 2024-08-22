/**** Contains implementation for a Multilayer Perceptron Neuron ****/

#ifndef MLP_NEURON_HPP
#define MLP_NEURON_HPP

#include <vector>

class MLPNeuron {
private:
    std::vector<float> prev_layer_inputs; // an array of neurons in the previous layer
    std::vector<float> respective_weights; // an array of weights associated with each neuron, respectively
    float bias;
    float (*activation_function) (float);

public:
    MLPNeuron(std::vector<float> prev_layer_inputs_arg, std::vector<float> respective_weights_arg,
              float bias_arg, float (*activation_function_arg) (float));

    float output();
};

MLPNeuron::MLPNeuron(std::vector<float> prev_layer_inputs_arg, std::vector<float> respective_weights_arg,
                     float bias_arg, float (*activation_function_arg) (float)) {

    prev_layer_inputs = prev_layer_inputs_arg;
    respective_weights = respective_weights_arg;
    bias = bias_arg;
    activation_function = activation_function_arg;
}

float MLPNeuron::output() {
    float weightedSum = bias;

    for (std::size_t i = 0; i < prev_layer_inputs.size(); ++i)
        weightedSum += prev_layer_inputs[i] * respective_weights[i];

    return activation_function(weightedSum);
}

#endif