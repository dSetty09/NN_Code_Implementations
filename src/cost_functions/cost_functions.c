#include "../../include/cost_functions/cost_functions.h"

float se(float y_m, float y_hat_m, int deriv) {
    return (deriv) ? -2 * (y_m - y_hat_m) : powf(y_m - y_hat_m, 2);
}

float multiclass_ce(float* y_hat_pdistro, int true_cls, int deriv) {
    return (deriv) ? - 1 / y_hat_pdistro[true_cls] : - logf(y_hat_pdistro[true_cls]);
}