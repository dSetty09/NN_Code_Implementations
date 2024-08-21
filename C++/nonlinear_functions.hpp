/* Notes:
 * --> Functions assume primitive integer data type passed as argument
 */

#ifndef NONLINEAR_FUNCTIONS_H
#define NONLINEAR_FUNCTIONS_H

#include <cmath>

// Sigmoid Activation Function
// --> Used for models where have to predict probability as output
long double sigmoid(long double x) {
    return 1 / (1 + std::exp(-x));
}

// Hyperbolic Tangent (tanh) Function
// --> Alternative to sigmoid function, except range is from
//     -1 to 1
long double hyperbolic_tangent(long double x) {
    long double e_neg_x = std::exp(-x);
    long double e_pos_x = std::exp(x);

    return (e_pos_x - e_neg_x) / (e_pos_x + e_neg_x);
}

// Step Function
// --> Used in binary classification
long double step(long double x) {
    return (x > 0) ? 1 : 0;
}

// Rectified Linear Unit (ReLU) Activation Function
// --> Similar to identity function except values less than
//     0 are mapped to 0
long double relu(long double x) {
    return (x > 0) ? x : 0;
}

// Leaky Rectified Linear Unit Activation Function
// --> Like ReLU except designed to handle the case where
//     weighted sum is less than 0. 
long double leaky_relu(long double x) {
    return (x >= 0) ? x : 0.01 * x;
}

// SoftPlus Activation Function
// --> Smoother approximation of ReLU
long double softplus(long double x) {
    return std::log(1 + std::exp(x));
}

// SoftMax Activation Function
// --> Applied in multi-class classification and used
//     to determine the probability of a weighted sum value
//     being the possible true output for a given input
// ---> There are n classes
long double* softmax(long double* v, unsigned int n) {
    long double denominator = 0;

    for (int j = 0; j < n; ++j) {
        denominator += std::exp(v[j]);
    }

    for (int i = 0; i < n; ++i) {
        v[i] = std::exp(v[i]) / denominator;
    }

    return v;
}

#endif
