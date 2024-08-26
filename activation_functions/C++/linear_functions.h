/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 */

#ifndef LINEAR_FUNCTIONS_HPP
#define LINEAR_FUNCTIONS_HPP

// Linear Activation Function
float linear(float x, int deriv) {
    return (deriv) ? 1 : x;
}

#endif