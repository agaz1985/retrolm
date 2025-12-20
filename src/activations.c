/**
 * @file activations.c
 * @brief Implementation of activation functions for neural networks
 */

#include "activations.h"
#include "matrix.h"

/**
 * @brief Apply ReLU activation function
 * 
 * Implementation using matrix clamp_min operation for efficiency.
 */
struct Matrix2D relu(const struct Matrix2D *m) {
	return mat_clamp_min(m, 0.0);
}

/**
 * @brief Apply Softmax activation function with numerical stability
 * 
 * Algorithm:
 * 1. Find max values per row for numerical stability
 * 2. Subtract max from input to prevent overflow
 * 3. Compute exponentials
 * 4. Sum exponentials per row
 * 5. Divide by sum to get normalized probabilities
 * 
 * All intermediate matrices are properly freed to prevent memory leaks.
 */
struct Matrix2D softmax(const struct Matrix2D *m) {
    struct Matrix2D max_vals = mat_max(m, 1);        // Max per row
    struct Matrix2D m_shifted = mat_sub(m, &max_vals); // Shift for stability
    struct Matrix2D num = mat_exp(&m_shifted);       // Compute exp(x - max)
    struct Matrix2D den = mat_sum(&num, 1);          // Sum per row
    struct Matrix2D result = mat_div(&num, &den);    // Normalize

    // Cleanup intermediate matrices
    mat_free(&max_vals);
    mat_free(&m_shifted);
    mat_free(&num);
    mat_free(&den);
    
    return result;
}