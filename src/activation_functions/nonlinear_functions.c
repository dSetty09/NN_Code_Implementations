#include "../../include/activation_functions/nonlinear_functions.h"

float sigmoid(float x, int deriv) {
    if (x < SIG_MIN_INPUT) x = SIG_MIN_INPUT;
    if (x > SIG_MAX_INPUT) x = SIG_MAX_INPUT;

    float e_neg_x = expf(-x);

    return (deriv) ? (e_neg_x / powf(1 + e_neg_x, 2)) + NEAR_ZERO : 1 / (1 + e_neg_x);
}

float hyperbolic_tangent(float x, int deriv) {
    if (x < TANH_MIN_INPUT) x = TANH_MIN_INPUT;
    if (x > TANH_MAX_INPUT) x = TANH_MAX_INPUT;

    float exp_sum = expf(x) + expf(-x);
    float exp_diff = expf(x) - expf(-x);

    if (deriv) { 
        float exp_sum_squared = powf(exp_sum, 2); 
        float exp_diff_squared = powf(exp_diff, 2);

        return ((exp_sum_squared - exp_diff_squared) / exp_sum_squared) + NEAR_ZERO;
    }

    return exp_diff / exp_sum;
}

float step(float x, int deriv) {
    if (deriv) return NEAR_ZERO;
    return (x > 0) ? 1 : 0;
}

float relu(float x, int deriv) {
    if (deriv) {
        if (x > 0) return 1 + NEAR_ZERO;
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

float softmax(float* z, int i, unsigned int K, int deriv_idx) {
    if (deriv_idx >= 0) {
        if (deriv_idx == i) {
            float sig_i = softmax(z, i, K, NO_DERIV);
            return (sig_i * (1 - sig_i)) + NEAR_ZERO;
        } else {
            return (-softmax(z, i, K, NO_DERIV) * softmax(z, deriv_idx, K, NO_DERIV)) + NEAR_ZERO;
        }
    }

    float sum_nat_exps = 0;

    for (int j = 0; j < K; ++j) sum_nat_exps += expf(z[j]);

    return expf(z[i]) / sum_nat_exps;
}
