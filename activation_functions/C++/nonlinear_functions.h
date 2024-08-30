/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 */

#ifndef NONLINEAR_FUNCTIONS_H
#define NONLINEAR_FUNCTIONS_H

#include <math.h>

// Sigmoid Activation Function
// --> Used for models where have to predict probability as output
float sigmoid(float x, int deriv) {
    if (deriv) {
        if (x <= -15) return 0;
        return expf(-x) / powf((1+expf(-x)), 2);
    }

    return 1 / (1 + expf(-x));
}

// Hyperbolic Tangent (tanh) Function
// --> Alternative to sigmoid function, except range is from
//     -1 to 1
float hyperbolic_tangent(float x, int deriv) {
    if (x <= -20) return (deriv) ? 0 : -1;
    if (x >= 20) return (deriv) ? 0 : 1;

    float e_neg_x = expf(-x);
    float e_pos_x = expf(x);

    if (deriv) {
        float sum_squared = powf(e_pos_x + e_neg_x, 2);
        float diff_squared = powf(e_pos_x - e_neg_x, 2);

        return (sum_squared - diff_squared) / sum_squared;
    }

    return (e_pos_x - e_neg_x) / (e_pos_x + e_neg_x);
}

// Step Function
// --> Used in binary classification
// --> Not useful for neural network learning as derivative
//     is equal to 0, except where x = 0 (as the derivative
//     doesn't exist at this point)
float step(float x, int deriv) {
    if (deriv) return (x != 0) ? 0 : NAN;
    return (x > 0) ? 1 : 0;
}

// Rectified Linear Unit (ReLU) Activation Function
// --> Similar to identity function except values less than
//     0 are mapped to 0
float relu(float x, int deriv) {
    if (deriv) {
        if (x == 0) return NAN;
        return (x > 0) ? 1 : 0;
    }

    return (x > 0) ? x : 0;
}

// Leaky Rectified Linear Unit Activation Function
// --> Like ReLU except designed to handle the case where
//     weighted sum is less than 0. 
float leaky_relu(float x, int deriv) {
    if (deriv) {
        if (x == 0) return NAN;
        return (x > 0) ? 1 : 0.01;
    }

    return (x >= 0) ? x : 0.01 * x;
}

// SoftPlus Activation Function
// --> Smoother approximation of ReLU
float softplus(float x, int deriv) {
    if (deriv) {
        if (x >= 30) return 1;
        return expf(x) / (1 + expf(x));
    }

    return log(1 + expf(x));

    return (deriv) ? expf(x) / (1 + expf(x)) : log(1 + expf(x));
}

// SoftMax Activation Function (TODO: fix this)
// --> Applied in multi-class classification and used
//     to determine the probability of a weighted sum value
//     being the possible true output for a given input
// ---> There are n classes
float* softmax(float* v, unsigned int n) {
    float denominator = 0;

    for (int j = 0; j < n; ++j) {
        denominator += expf(v[j]);
    }

    for (int i = 0; i < n; ++i) {
        v[i] = expf(v[i]) / denominator;
    }

    return v;
}

#endif
