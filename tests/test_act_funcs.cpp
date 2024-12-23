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
       leaky_relu_outputs = {0, -0.004, -0.16, 0.7, 22, -FLT_MAX * 0.01, FLT_MAX};
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

        if (deriv_flag && active_func != linear && active_func != leaky_relu) 
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

TEST(SoftMaxFuncTests, SoftMaxFuncTests) {
    float inputs[] = {12.45, 22.234, 7.765, -0.22234, -0.1245};
    float outputs[] = {0.0000563428, 0.9999431367, 0.0000005202, 0.0000000002, 0.0000000002};

    float deriv_outputs0[] = {0.0000563395823097, -0.0000563395529794, -2.9309373691e-11, -9.9574664328e-15, -1.0980957816e-14};
    float deriv_outputs1[] = {-0.0000563395529794, 0.0000568600927248, -5.2016814083e-7, -1.7672014614e-10, -1.9488456054e-10};
    float deriv_outputs2[] = {-2.9309373691e-11, -5.2016814083e-7, 5.201974504e-7, -9.1934644985e-17, -1.013842693e-16};
    float deriv_outputs3[] = {-9.9574664328e-15, -1.7672014614e-10, -9.1934644985e-17, 1.7673019557e-10, -3.4443945102e-20};
    float deriv_outputs4[] = {-1.0980957816e-14, -1.9488456054e-10, -1.013842693e-16, -3.4443945102e-20, 1.9489564292e-10};

    float* deriv_outputs[] = {deriv_outputs0, deriv_outputs1, deriv_outputs2, deriv_outputs3, deriv_outputs4};

    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(softmax(inputs, i, 5, NO_DERIV), outputs[i], 1e-5);

        for (int h = 0; h < 5; ++h) ASSERT_NEAR(softmax(inputs, i, 5, h), deriv_outputs[i][h], 1e-5);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}