/* TEST FILE FOR NEURAL NETWORK OPERATION FUNCTIONS */

#include <gtest/gtest.h>

#include "../include/neural_net_ops/neural_net_ops.h"

TEST(FloatSafeTests, Exp) {
    ASSERT_EQ(FLT_EPSILON, flt_safe_exp(-INFINITY));
    ASSERT_EQ(327747.843750, flt_safe_exp(12.7));
    ASSERT_EQ(FLT_MAX, flt_safe_exp(INFINITY));
}

TEST(FloatSafeTests, Log) {
    ASSERT_FLOAT_EQ(-103.278931, flt_safe_log(-INFINITY));
    ASSERT_FLOAT_EQ(62.169796, flt_safe_log(1e27)); 
    ASSERT_FLOAT_EQ(88.596848, flt_safe_log(INFINITY));
}

TEST(FloatSafeTests, Square) {
    ASSERT_FLOAT_EQ(FLT_MAX, flt_safe_square(-INFINITY));
    ASSERT_FLOAT_EQ(18044.279297, flt_safe_square(134.329)); 
    ASSERT_FLOAT_EQ(FLT_MAX, flt_safe_square(INFINITY));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}