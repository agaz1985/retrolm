/**
 * @file activations.h
 * @brief Activation functions for neural network computations
 * 
 * This module provides standard activation functions used in neural networks,
 * including ReLU and Softmax. These functions operate on 2D matrices and
 * create new matrix outputs without modifying the input.
 */

#ifndef _RLM_ACTIVATIONS_H
#define _RLM_ACTIVATIONS_H

/**
 * @brief Apply Rectified Linear Unit (ReLU) activation function
 * 
 * ReLU is defined as: f(x) = max(0, x)
 * This function clamps all negative values to zero while leaving positive values unchanged.
 * 
 * @param m Pointer to input matrix
 * @return New matrix with ReLU applied element-wise
 * 
 * @note The returned matrix must be freed by the caller using mat_free()
 * @note The input matrix is not modified
 */
struct Matrix2D relu(const struct Matrix2D *m);

/**
 * @brief Apply Softmax activation function along rows
 * 
 * Softmax converts logits to probability distributions by computing:
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * The function uses numerical stability by subtracting the max value before
 * computing exponentials to avoid overflow.
 * 
 * @param m Pointer to input matrix (typically logits)
 * @return New matrix where each row sums to 1.0 (probability distribution)
 * 
 * @note The returned matrix must be freed by the caller using mat_free()
 * @note The input matrix is not modified
 * @note Operates row-wise (each row becomes a probability distribution)
 */
struct Matrix2D softmax(const struct Matrix2D *m);

#endif // _RLM_ACTIVATIONS_H