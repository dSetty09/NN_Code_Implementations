#include "../../include/activation_functions/nonlinear_functions.h"

float sigmoid(float x, int deriv) {
    if (x < SIG_MIN_INPUT) x = SIG_MIN_INPUT;
    if (x > SIG_MAX_INPUT) x = SIG_MAX_INPUT;
    return (deriv) ? expf(-x) / powf(1 + expf(-x), 2) : 1 / (1 + expf(-x));
}

float hyperbolic_tangent(float x, int deriv) {
    float result;

    float e_neg_x = flt_safe_exp(-x);
    float e_pos_x = flt_safe_exp(x);

    if (deriv) {
        float sum_squared = flt_safe_square(e_pos_x + e_neg_x);
        float diff_squared = flt_safe_square(e_pos_x - e_neg_x);

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

    float e_pos_x = flt_safe_exp(x);
    float log_expr = 1 + e_pos_x;

    if (deriv) {
        result = e_pos_x / log_expr;
        return result + NEAR_ZERO;
    }

    result = flt_safe_log(1 + flt_safe_exp(x));
    return result;
}

float softmax(float* z, int i, unsigned int n, int deriv_i) {
    float sum_nat_exps = 0;

    for (int j = 0; j < n; ++j) sum_nat_exps += flt_safe_exp(z[j]);

    if (deriv_i >= 0) {
        if (i == deriv_i) {
            float sum_nat_exps_before = 0;
            for (int k = 0; k < i; ++k) sum_nat_exps_before += flt_safe_exp(z[k]);

            float sum_nat_exps_after = 0;
            for (int l = i + 1; l < n; ++l) sum_nat_exps_after += flt_safe_exp(z[l]);

            return (flt_safe_exp(z[i]) * (sum_nat_exps_before + sum_nat_exps_after)) / (sum_nat_exps * sum_nat_exps);

        }
        
        return -(flt_safe_exp(z[i] + z[deriv_i]) / (sum_nat_exps * sum_nat_exps));
    }

    return flt_safe_exp(z[i]) / sum_nat_exps;
}
