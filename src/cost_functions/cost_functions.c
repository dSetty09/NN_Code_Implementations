#include "../../include/cost_functions/cost_functions.h"

float se(float y_m, float y_hat_m, int deriv) {
    return (deriv) ? -2 * (y_m - y_hat_m) : powf(y_m - y_hat_m, 2);
}

float multiclass_ce(float* y_pdistro, float* y_hat_pdistro, int num_classes, int deriv_idx) {
    if (deriv_idx >= 0) return -y_pdistro[deriv_idx] / y_hat_pdistro[deriv_idx];

    float ret = 0;

    for (int i = 0; i < num_classes; ++i) ret -= y_pdistro[i] * logf(y_hat_pdistro[i]);

    return ret;
}

float mean_multiclass_ce(float** y_pdistros, float** y_hat_pdistros, int num_datapoints, int num_classes) {
    float ret = 0;

    for (int i = 0; i < num_datapoints; ++i) 
        ret += multiclass_ce(y_pdistros[i], y_hat_pdistros[i], num_classes, NO_DERIV); 

    ret /= num_datapoints;

    return ret;
}