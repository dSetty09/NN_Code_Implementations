/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 */

#ifndef LINEAR_FUNCTIONS_HPP
#define LINEAR_FUNCTIONS_HPP

#include "../../common_definitions.h"

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

#endif