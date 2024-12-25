#include "../../include/cost_functions/cost_functions.h"

float se(float y_m, float y_hat_m, int deriv) {
    return (deriv) ? -2 * (y_m - y_hat_m) : powf(y_m - y_hat_m, 2);
}

float cross_entropy(float* yielded_distro, float* true_distro, int n, int deriving_output_index) {
    return 1.0;
}