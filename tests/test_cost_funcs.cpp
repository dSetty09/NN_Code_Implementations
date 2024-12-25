/* TEST FILE FOR COST FUNCTIONS */

#include <gtest/gtest.h>

#include <vector>

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}