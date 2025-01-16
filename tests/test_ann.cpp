/* TEST FILE FOR ANN */

#include <gtest/gtest.h>

#include "../include/activation_functions/nonlinear_functions.h"
#include "../include/cost_functions/cost_functions.h"
#include "../include/unit/unit.h"
#include "../include/ann/dense_layer.h"
#include "../include/ann/ann.h"

class NNLayerTests : public testing::Test {
protected:
    void SetUp() override {
        dense_h1_layer = init_dense_layer(2, (Neuron) {.weights=WEIGHTS(-0.795, 0.588, -0.704), .num_weights=3,
                                                       .bias=0.864, .act_func=(void*) leaky_relu}, 
                                             (Neuron) {.weights=WEIGHTS(-0.603, -0.204, 0.921), .num_weights=3,
                                                       .bias=-0.698, .act_func=(void*) leaky_relu}); 

        dense_h2_layer = init_dense_layer(3, (Neuron) {.weights=WEIGHTS(-0.181, 0.078), .num_weights=2,
                                                       .bias=0.248, .act_func=(void*) leaky_relu}, 
                                             (Neuron) {.weights=WEIGHTS(0.012, 0.034), .num_weights=2,
                                                       .bias=0.240, .act_func=(void*) leaky_relu},
                                             (Neuron) {.weights=WEIGHTS(-0.461, 0.424), .num_weights=2,
                                                       .bias=0.207, .act_func=(void*) leaky_relu});

        dense_output_layer = init_dense_layer(2, (Neuron) {.weights=WEIGHTS(0.234, 0.865, -0.125), .num_weights=3,
                                                           .bias=-0.877, .act_func=(void*) softmax}, 
                                                 (Neuron) {.weights=WEIGHTS(0.678, 0.958, -0.485), .num_weights=3,
                                                           .bias=0.698, .act_func=(void*) softmax});

        ann = init_ann(MULTI_CLASS_CROSS_ENTROPY, 3, dense_h1_layer, dense_h2_layer, dense_output_layer);

        x = (float[]) {1, 2, 3};
    }

    void TearDown() override {
        del_ann(ann);
    }

    float* x;

    DenseLayer dense_h1_layer;
    DenseLayer dense_h2_layer;
    DenseLayer dense_output_layer;
    ArtificialNeuralNetwork* ann;
};

TEST_F(NNLayerTests, Creation) {
    int num_layers = 3;
    int num_neurons_per_layer[3] = {2, 3, 2};

    float w[3][3][3] = {{{-0.795, 0.588, -0.704}, {-0.603, -0.204, 0.921}, {0, 0, 0}}, 
                        {{-0.181, 0.078, 0}, {0.012, 0.034, 0}, {-0.461, 0.424, 0}}, 
                        {{0.234, 0.865, -0.125}, {0.678, 0.958, -0.485}, {0, 0, 0}}};

    int num_weights_per_layer[3] = {3, 2, 3};

    float b[3][3] = {{0.864, -0.698, 0}, {0.248, 0.240, 0.207}, {-0.877, 0.698, 0}};

    int num_b_per_layer[3] = {2, 3, 2};

    void* act_funcs_per_layer[3] = {(void*) leaky_relu, (void*) leaky_relu, (void*) softmax};

    ASSERT_EQ(NNLayerTests::ann->num_layers, 3);
    ASSERT_EQ(NNLayerTests::ann->cost_function, (void*) multiclass_ce);

    for (int i = 0; i < num_layers; ++i) {
        ASSERT_EQ(NNLayerTests::ann->layers[i].num_neurons, num_neurons_per_layer[i]);

        for (int j = 0; j < num_neurons_per_layer[i]; ++j) {
            NeuronNode curr_neuron = NNLayerTests::ann->layers[i].neurons[j];

            ASSERT_EQ(curr_neuron.num_weights, num_weights_per_layer[i]);

            for (int k = 0; k < num_weights_per_layer[i]; ++k) ASSERT_FLOAT_EQ(curr_neuron.w[k], w[i][j][k]);
            ASSERT_FLOAT_EQ(curr_neuron.b, b[i][j]);

            for (int k = 0; k < num_weights_per_layer[i]; ++k) ASSERT_FLOAT_EQ(curr_neuron.delta_w[k], 0);
            ASSERT_FLOAT_EQ(curr_neuron.delta_b, 0);

            ASSERT_EQ(curr_neuron.act_func, act_funcs_per_layer[i]);
            ASSERT_FLOAT_EQ(curr_neuron.deriv_a, 0);

            ASSERT_FLOAT_EQ(curr_neuron.output, 0);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}