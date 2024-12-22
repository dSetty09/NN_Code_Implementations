/* TEST FILE FOR NEURAL NETWORK OPERATION FUNCTIONS */

#include <gtest/gtest.h>

#include "../include/neural_net_ops/neural_net_ops.h"

TEST(BackPropSafeTests, Exp) {
    float regular = 12.7;
    float too_large = INFINITY;

    ASSERT_NEAR(0, bp_safe_exp(-INFINITY), 1e-5);
    ASSERT_NEAR(327747.843750, bp_safe_exp(12.7), 1e-5);
    ASSERT_NEAR(FLT_MAX, bp_safe_exp(INFINITY), 1e-5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}