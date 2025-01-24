/* TEST FILE FOR COST FUNCTIONS */

#include <gtest/gtest.h>

#include "../include/cost_functions/cost_functions.h"

TEST(SquaredErrorTests, ZeroError) {
    ASSERT_EQ(se(3, 3, 0), 0);
}

TEST(SquaredErrorDerivTests, ZeroErrorDeriv) {
    ASSERT_EQ(se(3, 3, 1), 0);
}

TEST(SquaredErrorTests, SmallError) {
    ASSERT_FLOAT_EQ(se(3, 6, 0), 9);
    ASSERT_FLOAT_EQ(se(6, 3, 0), 9);
}

TEST(SquaredErrorDerivTests, SmallErrorDeriv) {
    ASSERT_FLOAT_EQ(se(3, 6, 1), 6);
    ASSERT_FLOAT_EQ(se(6, 3, 1), -6);
}

TEST(SquaredErrorTests, LargeError) {
    ASSERT_FLOAT_EQ(se(3, 103, 0), 10000);
    ASSERT_FLOAT_EQ(se(103, 3, 0), 10000);
}

TEST(SquaredErrorDerivTests, LargeErrorDeriv) {
    ASSERT_FLOAT_EQ(se(3, 103, 1), 200);
    ASSERT_FLOAT_EQ(se(103, 3, 1), -200);
}

class MultiClassCrossEntropyTests : public testing::Test {
protected:
    void SetUp() override {
        uncertain_pdistro[0] = 0.1; uncertain_pdistro[1] = 0.2; uncertain_pdistro[2] = 0.7;
        certain_pdistro[0] = 1e-5; certain_pdistro[1] = 1e-5; certain_pdistro[2] = 1 - 2e-5;
    }

    float uncertain_pdistro[3]; 
    float certain_pdistro[3];
};

TEST_F(MultiClassCrossEntropyTests, ZeroUncertainty) {
    float true_pdistro[] = {0, 0, 1};
    ASSERT_NEAR(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, NO_DERIV), 0, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, ZeroUncertaintyDeriv) {
    float true_pdistro[] = {0, 0, 1};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 0), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 1), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 2), -1.0000200004);
}

TEST_F(MultiClassCrossEntropyTests, HighUncertainty) {
    float true_pdistro[] = {1, 0, 0};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, NO_DERIV), 
                    11.512925465);
}

TEST_F(MultiClassCrossEntropyTests, HighUncertaintyDeriv) {
    float true_pdistro[] = {1, 0, 0};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 0), -100000);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 1), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::certain_pdistro, 3, 2), 0);
}

TEST_F(MultiClassCrossEntropyTests, LittleUncertainty) {
    float true_pdistro[] = {0, 0, 1};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, NO_DERIV), 
                    0.356674943939);
}

TEST_F(MultiClassCrossEntropyTests, LittleUncertaintyDeriv) {
    float true_pdistro[] = {0, 0, 1};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 0), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 1), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 2), 
                    -1.42857142857);
}

TEST_F(MultiClassCrossEntropyTests, LargeUncertainty) {
    float true_pdistro[] = {1, 0, 0};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, NO_DERIV), 
                    2.30258509299);
}

TEST_F(MultiClassCrossEntropyTests, LargeUncertaintyDeriv) {
    float true_pdistro[] = {1, 0, 0};
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 0), -10);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 1), 0);
    ASSERT_FLOAT_EQ(multiclass_ce(true_pdistro, MultiClassCrossEntropyTests::uncertain_pdistro, 3, 2), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}