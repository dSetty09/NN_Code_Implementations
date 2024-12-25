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
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::certain_pdistro, 2, 0), 0, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, ZeroUncertaintyDeriv) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::certain_pdistro, 2, 1), -1, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, MaxUncertainty) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::certain_pdistro, 1, 0), 11.512925, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, MaxUncertaintyDeriv) {
    ASSERT_LT(multiclass_ce(MultiClassCrossEntropyTests::certain_pdistro, 1, 1), -1000);
}

TEST_F(MultiClassCrossEntropyTests, LittleUncertainty) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::uncertain_pdistro, 2, 0), 0.356675, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, LittleUncertaintyDeriv) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::uncertain_pdistro, 2, 1), -1.428571, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, LargeUncertainty) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::uncertain_pdistro, 0, 0), 2.302585, 1e-4);
}

TEST_F(MultiClassCrossEntropyTests, LargeUncertaintyDeriv) {
    ASSERT_NEAR(multiclass_ce(MultiClassCrossEntropyTests::uncertain_pdistro, 0, 1), -10, 1e-4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}