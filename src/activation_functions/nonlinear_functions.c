#include "../../include/activation_functions/nonlinear_functions.h"

float sigmoid(float x, int deriv) {
    float result;

    float e_neg_x = bp_safe_exp(-x, 0);
    float sig_denom = 1 + bp_safe_exp(-x, 0);

    if (deriv) {
        if (x < -35) {
            return NEAR_ZERO;
        }

        float sig_denom_squared = bp_safe_square(sig_denom, 0);

        result = e_neg_x / sig_denom_squared;
        return result + NEAR_ZERO;
    }

    result = 1 / sig_denom;
    return result;
}

float hyperbolic_tangent(float x, int deriv) {
    float result;

    float e_neg_x = bp_safe_exp(-x, 0);
    float e_pos_x = bp_safe_exp(x, 0);

    if (deriv) {
        float sum_squared = bp_safe_square(e_pos_x + e_neg_x, 0);
        float diff_squared = bp_safe_square(e_pos_x - e_neg_x, 0);

        result = (sum_squared - diff_squared) / sum_squared;
        return result + NEAR_ZERO;
    }

    result = (e_pos_x - e_neg_x) / (e_pos_x + e_neg_x);
    return result;
}

float step(float x, int deriv) {
    if (deriv) return NEAR_ZERO;
    return (x > 0) ? 1 : 0;
}

float relu(float x, int deriv) {
    if (deriv) {
        if (x > 0) {
            return 1;
        } 

        return NEAR_ZERO;
    }

    if (x < 0) {
        return 0;
    }

    if (x > FLT_MAX) {
        return FLT_MAX;
    } 

    return x;
}

float leaky_relu(float x, int deriv) {
    if (deriv) {
        if (x > 0) {
            return 1;
        }

        return 0.01;
    }

    if (x < 0) {
        if (x < -FLT_MAX) {
            return 0.01 * -FLT_MAX;
        }

        return 0.01 * x;
    }

    if (x > FLT_MAX) {
        return FLT_MAX;
    }

    return x;
}

float softplus(float x, int deriv) {
    if (x > FLT_MAX) {
        if (deriv) {
            return 1;
        }

        return FLT_MAX;
    }

    float result;

    float e_pos_x = bp_safe_exp(x, 0);
    float log_expr = 1 + e_pos_x;

    if (deriv) {
        result = e_pos_x / log_expr;
        return result + NEAR_ZERO;
    }

    result = bp_safe_log(1 + bp_safe_exp(x, 0), 0);
    return result;
}

float softmax(float* z, int i, unsigned int n, int deriv_i) {
    float sum_nat_exps = 0;

    for (int j = 0; j < n; ++j) sum_nat_exps += bp_safe_exp(z[j], 0);

    if (deriv_i >= 0) {
        if (i == deriv_i) {
            float sum_nat_exps_before = 0;
            for (int k = 0; k < i; ++k) sum_nat_exps_before += bp_safe_exp(z[k], 0);

            float sum_nat_exps_after = 0;
            for (int l = i + 1; l < n; ++l) sum_nat_exps_after += bp_safe_exp(z[l], 0);

            return (bp_safe_exp(z[i], 0) * (sum_nat_exps_before + sum_nat_exps_after)) / (sum_nat_exps * sum_nat_exps);

        }
        
        return -(bp_safe_exp(z[i] + z[deriv_i], 0) / (sum_nat_exps * sum_nat_exps));
    }

    return bp_safe_exp(z[i], 0) / sum_nat_exps;
}
