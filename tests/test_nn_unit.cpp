/* TEST FILE FOR NEURAL NETWORK UNITS, SUCH AS A NEURON OR KERNEL */

#include <gtest/gtest.h>

#include "../include/activation_functions/nonlinear_functions.h"
#include "../include/unit/unit.h"

class NeuronUnitTests : public testing::Test {
protected:
    void SetUp() override {
        neuron = init_neuron({.act_func=(void*) relu, .num_weights=5, .bias=0.877, 
                              WEIGHTS(0.632, 0.571, 0.991, 0.529, 0.492)});

        x = (float[]) {1, 2, 3, 4, 5};
    }

    void TearDown() override {
        del_neuron(neuron);
    }

    float* x;

    NeuronNode neuron;
};

TEST_F(NeuronUnitTests, Creation) {
    float w[] = {0.632, 0.571, 0.991, 0.529, 0.492};
    float num_weights = 5;
    float b = 0.877;

    for (int i = 0; i < 5; ++i) ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.w[i], w[i]);
    ASSERT_EQ(NeuronUnitTests::neuron.num_weights, num_weights); 

    ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.b, b);

    for (int i = 0; i < 5; ++i) ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.delta_w[i], 0);
    ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.delta_b, 0);
    ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.deriv_a, 0);

    ASSERT_FLOAT_EQ(NeuronUnitTests::neuron.output, 0);

    ASSERT_EQ(NeuronUnitTests::neuron.act_func, (void*) relu);
}

TEST_F(NeuronUnitTests, WeightedSum) {
    ASSERT_FLOAT_EQ(wsum(&(NeuronUnitTests::neuron), NeuronUnitTests::x, NO_DERIV, 0, 0, 0), 10.2);
}

TEST_F(NeuronUnitTests, WeightDeriv) {
    ASSERT_EQ(wsum(&(NeuronUnitTests::neuron), NeuronUnitTests::x, 2, TRUE, FALSE, FALSE), 3);
}

TEST_F(NeuronUnitTests, InputDeriv) {
    ASSERT_FLOAT_EQ(wsum(&(NeuronUnitTests::neuron), NeuronUnitTests::x, 2, FALSE, TRUE, FALSE), 0.991);
}

TEST_F(NeuronUnitTests, BiasDeriv) {
    ASSERT_EQ(wsum(&(NeuronUnitTests::neuron), NeuronUnitTests::x, NO_DERIV, FALSE, FALSE, TRUE), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}