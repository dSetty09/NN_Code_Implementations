/* TEST FILE FOR NEURAL NETWORK UNITS, SUCH AS A NEURON OR KERNEL */

#include <gtest/gtest.h>

#include "../include/activation_functions/nonlinear_functions.h"
#include "../include/unit/unit.h"

class NeuronUnitTests : public testing::Test {
protected:
    void SetUp() override {
        w[0] = 0.632; w[1] = 0.571; w[2] = 0.991; w[3] = 0.529; w[4] = 0.492;
        b = 0.877;
        x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4; x[4] = 5;

        neuron = (Neuron*) malloc(sizeof(Neuron));

        neuron->w = NULL;
        neuron->W = 5;
        neuron->b = b;
        neuron->delta_w = NULL;
        neuron->delta_b = 0;
        neuron->act_func = relu;
        neuron->soft_act_func = NULL;

        arr_dup(NeuronUnitTests::w, &(neuron->w), 5);

        neuron->delta_w = (float*) calloc(sizeof(float), 5);    
        neuron->delta_x = (float*) calloc(sizeof(float), 5);
    }

    void TearDown() override {
        arr_delete(neuron->w);
        arr_delete(neuron->delta_w);
        arr_delete(neuron->delta_x);

        free(neuron);
    }

    float w[5];
    float x[5];
    float b;

    Neuron* neuron;
};

TEST_F(NeuronUnitTests, WeightedSum) {
    ASSERT_FLOAT_EQ(wsum(NeuronUnitTests::neuron, NeuronUnitTests::x, NO_DERIV, 0, 0, 0), 10.2);
}

TEST_F(NeuronUnitTests, WeightDeriv) {
    ASSERT_EQ(wsum(NeuronUnitTests::neuron, NeuronUnitTests::x, 2, 1, 0, 0), 3);
}

TEST_F(NeuronUnitTests, InputDeriv) {
    ASSERT_FLOAT_EQ(wsum(NeuronUnitTests::neuron, NeuronUnitTests::x, 2, 0, 1, 0), 0.991);
}

TEST_F(NeuronUnitTests, BiasDeriv) {
    ASSERT_EQ(wsum(NeuronUnitTests::neuron, NeuronUnitTests::x, NO_DERIV, 0, 0, 1), 1);
}

TEST_F(NeuronUnitTests, Gradients) {
    float cost_to_wsum_deriv = 2;

    update_gradients(NeuronUnitTests::neuron, NeuronUnitTests::x, cost_to_wsum_deriv);

    for (int i = 0; i < 5; ++i) ASSERT_FLOAT_EQ(NeuronUnitTests::neuron->delta_w[i], 2 * x[i]);
    for (int i = 0; i < 5; ++i) ASSERT_FLOAT_EQ(NeuronUnitTests::neuron->delta_x[i], 2 * w[i]);
    ASSERT_FLOAT_EQ(NeuronUnitTests::neuron->delta_b, 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}