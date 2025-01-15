/* TEST FILE FOR DENSE LAYER OF ARTIFICIAL NEURAL NETWORK */

#include <gtest/gtest.h>

#include "../include/activation_functions/nonlinear_functions.h"

#include "../include/ann/dense_layer.h"
#include "../include/unit/unit.h"

class DenseLayerTests : public testing::Test {
protected:
    void SetUp() override {
        dense_norm_layer = init_dense_layer(3, (Neuron) {.weights=WEIGHTS(0.234, 0.865, 0.125), .num_weights=3,
                                                         .bias=0.877, .act_func=(void*) hyperbolic_tangent}, 
                                               (Neuron) {.weights=WEIGHTS(0.678, 0.958, 0.485), .num_weights=3,
                                                         .bias=0.698, .act_func=(void*) hyperbolic_tangent}, 
                                               (Neuron) {.weights=WEIGHTS(0.546, 0.062, 0.498), .num_weights=3,
                                                         .bias=0.346, .act_func=(void*) hyperbolic_tangent}); 

        dense_output_layer = init_dense_layer(3, (Neuron) {.weights=WEIGHTS(0.234, 0.865, 0.125), .num_weights=3,
                                                           .bias=0.877, .act_func=(void*) softmax}, 
                                                 (Neuron) {.weights=WEIGHTS(0.678, 0.958, 0.485), .num_weights=3,
                                                           .bias=0.698, .act_func=(void*) softmax}, 
                                                 (Neuron) {.weights=WEIGHTS(0.546, 0.062, 0.498), .num_weights=3,
                                                           .bias=0.346, .act_func=(void*) softmax});

        x = (float[]) {1, 2, 3};
    }

    void TearDown() override {
        del_dense_layer(dense_norm_layer);
        del_dense_layer(dense_output_layer);
    }

    float* x;
    DenseLayer dense_norm_layer; 
    DenseLayer dense_output_layer;
};

TEST_F(DenseLayerTests, Creation) {
    ASSERT_EQ(DenseLayerTests::dense_norm_layer.num_neurons, 3);

    float w[3][3] = {{0.234, 0.865, 0.125}, {0.678, 0.958, 0.485}, {0.546, 0.062, 0.498}};
    int num_weights = 3;

    float b[3] = {0.877, 0.698, 0.346};

    for (int i = 0; i < DenseLayerTests::dense_norm_layer.num_neurons; ++i) {
        NeuronNode curr_neuron = DenseLayerTests::dense_norm_layer.neurons[i];

        for (int j = 0; j < 3; ++j) ASSERT_FLOAT_EQ(curr_neuron.w[j], w[i][j]);

        ASSERT_FLOAT_EQ(curr_neuron.b, b[i]);

        for (int j = 0; j < 3; ++j) ASSERT_FLOAT_EQ(curr_neuron.delta_w[i], 0);
        ASSERT_FLOAT_EQ(curr_neuron.delta_b, 0);
        ASSERT_FLOAT_EQ(curr_neuron.deriv_a, 0);

        ASSERT_FLOAT_EQ(curr_neuron.output, 0);

        ASSERT_EQ(curr_neuron.act_func, (void*) hyperbolic_tangent);
    }
}

TEST_F(DenseLayerTests, NormalForwardPass) {
    float expected_outputs[3] = {0.996786712605, 0.999849406758, 0.986877613732}; 

    float* outputs = NULL;
    DenseLayer* dl = &(DenseLayerTests::dense_norm_layer);

    start_forward_pass(dl, &outputs);

    conduct_forward_pass(dl->neurons, dl->num_neurons, DenseLayerTests::x, FALSE, outputs);
    for (int i = 0; i < 3; ++i) ASSERT_FLOAT_EQ(expected_outputs[i], outputs[i]);
    
    end_forward_pass(outputs);
}

TEST_F(DenseLayerTests, OutputLayerForwardPass) {
    float expected_outputs[3] = {0.163494544917, 0.755802142274, 0.0807033128095}; 

    float* outputs = NULL;
    DenseLayer* dl = &(DenseLayerTests::dense_output_layer);

    start_forward_pass(dl, &outputs);

    conduct_forward_pass(dl->neurons, dl->num_neurons, DenseLayerTests::x, TRUE, outputs);
    for (int i = 0; i < 3; ++i) ASSERT_FLOAT_EQ(expected_outputs[i], outputs[i]);
    
    end_forward_pass(outputs);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}