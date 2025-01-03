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

TEST(RandomNumGeneratorTests, SmallRange) {
    int randn = (int) random_num(0, 5.5, 0);

    ASSERT_GE(randn, 0);
    ASSERT_LT(randn, 5.5);
}

TEST(RandomSampleTests, SampleMany) {
    int* sample = sample_indices(50, 1000);

    int* observed = (int*) calloc(1000, sizeof(int));

    for (int i = 0; i < 50; ++i) {
        ASSERT_EQ(observed[sample[i]], FALSE);
        observed[sample[i]] = TRUE;

        ASSERT_GE(sample[i], 0);
        ASSERT_LT(sample[i], 1000);
   }

    free(sample);
}

TEST(BatchesTests, Standard) {
    int num_batches = 0;
    int last_batch_size = 0;
    int** batches = generate_training_batches(110, 20, &num_batches, &last_batch_size);

    int* observed = (int*) calloc(110, sizeof(int));
    for (int i = 0; i < 110; ++i) observed[i] = FALSE; 

    int num_unique_observations = 0;

    for (int b = 0; b < 6; ++b) {
        if (b < 5) {
            for (int i = 0; i < 20; ++i) {
                ASSERT_EQ(observed[batches[b][i]], FALSE);
                observed[batches[b][i]] = TRUE;

                ++num_unique_observations;
            }
        } else {
            for (int i = 0; i < 10; ++i) {
                ASSERT_EQ(observed[batches[b][i]], FALSE);
                observed[batches[b][i]] = TRUE;

                ++num_unique_observations;
            } 
        }
    }

    ASSERT_EQ(num_unique_observations, 110);

    free(observed);

    for (int b = 0; b < 6; ++b) free(batches[b]);
    free(batches);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}