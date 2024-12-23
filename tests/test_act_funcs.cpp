/* TEST FILE FOR ACTIVATION FUNCTIONS */

#include <gtest/gtest.h>

#include <vector>

#include "../include/neural_net_ops/neural_net_ops.h"
#include "../include/activation_functions/linear_functions.h"
#include "../include/activation_functions/nonlinear_functions.h"

class SimpleActFuncTest : public testing::Test {
protected:
    void SetUp() override {
       inputs = {0, -0.4, -16, 0.7, 22, -INFINITY, INFINITY}; 

       linear_outputs = {0, -0.4, -16, 0.7, 22, -FLT_MAX, FLT_MAX};
       sigmoid_outputs = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};
       tanh_outputs = {0, -0.37995, -1, 0.60437, 1, -1, 1};
       step_outputs = {0, 0, 0, 1, 1, 0, 1};
       relu_outputs = {0, 0, 0, 0.7, 22, 0, FLT_MAX};
       leaky_relu_outputs = {0, -0.004, -0.16, 0.7, 22, -FLT_MAX, FLT_MAX};
       softplus_outputs = {0.69315, 0.51302, 0, 1.10319, 22, 0, FLT_MAX};

       linear_deriv_outputs = {1, 1, 1, 1, 1, 1, 1};
       sigmoid_deriv_outputs = {0.25, 0.24026, 0, 0.22171, 0, 0, 0};
       tanh_deriv_outputs = {1, 0.85564, 0, 0.63474, 0, 0, 0};
       step_deriv_outputs = {0, 0, 0, 0, 0, 0, 0};
       relu_deriv_outputs = {0, 0, 0, 1, 1, 0, 1};
       leaky_relu_deriv_outputs = {0.01, 0.01, 0.01, 1, 1, 0.01, 1};
       softplus_deriv_outputs = {0.5, 0.40131, 0, 0.66819, 1, 0, 1};

       num_tests = 7;

       no_deriv = 0;
       deriv = 1;
    }

    std::vector<float> inputs, linear_outputs, sigmoid_outputs,
                       tanh_outputs, step_outputs, relu_outputs,
                       leaky_relu_outputs, softplus_outputs, 
                       linear_deriv_outputs, sigmoid_deriv_outputs,
                       tanh_deriv_outputs, step_deriv_outputs, 
                       relu_deriv_outputs, leaky_relu_deriv_outputs,
                       softplus_deriv_outputs;

    int num_tests;

    int no_deriv;
    int deriv;
};

void eval_simple_act_func(simple_act_func active_func, int deriv_flag, int num_tests, 
                          const std::vector<float>& inputs, const std::vector<float>& outputs) {

    for (int i = 0; i < num_tests; ++i) {
        ASSERT_NEAR(active_func(inputs[i], deriv_flag), outputs[i], 1e-5); 

        if (deriv_flag && active_func != linear) 
            ASSERT_FALSE(are_equal(active_func(inputs[i], deriv_flag), outputs[i]));
    }
}

TEST_F(SimpleActFuncTest, LinearFuncTest) {
    eval_simple_act_func(linear, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::linear_outputs);
}

TEST_F(SimpleActFuncTest, SigmoidFuncTest) {
    eval_simple_act_func(sigmoid, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::sigmoid_outputs);
}

TEST_F(SimpleActFuncTest, TanhFuncTest) {
    eval_simple_act_func(hyperbolic_tangent, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::tanh_outputs);
}

TEST_F(SimpleActFuncTest, StepFuncTest) {
    eval_simple_act_func(step, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::step_outputs);
}

TEST_F(SimpleActFuncTest, ReluFuncTest) {
    eval_simple_act_func(relu, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::relu_outputs);
}

TEST_F(SimpleActFuncTest, LeakyReluFuncTest) {
    eval_simple_act_func(leaky_relu, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::leaky_relu_outputs);
}

TEST_F(SimpleActFuncTest, SoftPlusFuncTest) {
    eval_simple_act_func(softplus, SimpleActFuncTest::no_deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::softplus_outputs);
}

TEST_F(SimpleActFuncTest, LinearDerivFuncTest) {
    eval_simple_act_func(linear, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::linear_deriv_outputs);
}

TEST_F(SimpleActFuncTest, SigmoidDerivFuncTest) {
    eval_simple_act_func(sigmoid, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::sigmoid_deriv_outputs);
}

TEST_F(SimpleActFuncTest, TanhDerivFuncTest) {
    eval_simple_act_func(hyperbolic_tangent, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::tanh_deriv_outputs);
}

TEST_F(SimpleActFuncTest, StepDerivFuncTest) {
    eval_simple_act_func(step, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::step_deriv_outputs);
}

TEST_F(SimpleActFuncTest, ReluDerivFuncTest) {
    eval_simple_act_func(relu, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::relu_deriv_outputs);
}

TEST_F(SimpleActFuncTest, LeakyReluDerivFuncTest) {
    eval_simple_act_func(leaky_relu, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::leaky_relu_deriv_outputs);
}

TEST_F(SimpleActFuncTest, SoftPlusDerivFuncTest) {
    eval_simple_act_func(softplus, SimpleActFuncTest::deriv, SimpleActFuncTest::num_tests,
                         SimpleActFuncTest::inputs, SimpleActFuncTest::softplus_deriv_outputs);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}