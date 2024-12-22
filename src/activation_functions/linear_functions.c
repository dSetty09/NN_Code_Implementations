#include "../../include/activation_functions/linear_functions.h"

// Linear Activation Function
float linear(float x, int deriv) {
    if (deriv) {
        return 1;
    }

    if (x > FLT_MAX) {
        return FLT_MAX;
    }

    if (x < -FLT_MAX) {
        return -FLT_MAX;
    }

    return x;
}