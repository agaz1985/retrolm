/**
 * @file layers.h
 * @brief Neural network layer implementations for transformer models
 * 
 * This module provides the core building blocks for transformer-based language models:
 * - Linear (fully connected) layers with bias
 * - Self-attention mechanisms
 * - Token embedding layers
 * 
 * Each layer type has its own parameter structure and associated functions for
 * creation, initialization, memory management, and forward propagation.
 */

#ifndef _RLM_LAYERS_H
#define _RLM_LAYERS_H

#include "matrix.h"

/**
 * @brief Parameters for a linear (fully connected) layer
 * 
 * A linear layer computes: y = xW^T + b
 * where x is the input, W is the weight matrix, and b is the bias vector.
 */
struct LinearParameters {
	struct Matrix2D weights;  /**< Weight matrix [out_features x in_features] */
	struct Matrix2D bias;     /**< Bias vector [1 x out_features] */
};

/**
 * @brief Parameters for a self-attention layer
 * 
 * Self-attention allows the model to weigh the importance of different positions
 * in a sequence. It uses four linear projections: Query (Q), Key (K), Value (V),
 * and Output (O).
 */
struct SelfAttentionParameters {
	struct LinearParameters Wq;  /**< Query projection */
	struct LinearParameters Wk;  /**< Key projection */
	struct LinearParameters Wv;  /**< Value projection */
	struct LinearParameters Wo;  /**< Output projection */
};

/**
 * @brief Parameters for an embedding layer
 * 
 * Embeddings convert discrete token indices into dense vector representations.
 * The weight matrix is indexed by token ID to retrieve the corresponding embedding.
 */
struct EmbeddingsParameters {
	struct Matrix2D weight_matrix;  /**< Embedding lookup table [vocab_size x embed_dim] */
};

/* ========================================
 * Linear Layer Functions
 * ========================================
 * A linear layer performs an affine transformation: y = xW^T + b
 */

/**
 * @brief Create a new linear layer with uninitialized parameters
 * 
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @return LinearParameters structure with allocated but uninitialized weights and bias
 * 
 * @note Weights will be [out_features x in_features], bias will be [1 x out_features]
 * @note Call linear_random_init() to initialize parameters with random values
 * @note Call linear_free() when done to release memory
 */
struct LinearParameters linear_new(unsigned int in_features, unsigned int out_features);

/**
 * @brief Create a deep copy of linear layer parameters
 * 
 * @param p Pointer to source LinearParameters to copy
 * @return New LinearParameters with copied weights and bias
 * 
 * @note The returned structure must be freed with linear_free()
 */
struct LinearParameters linear_copy(const struct LinearParameters *p);

/**
 * @brief Initialize linear layer parameters with random values
 * 
 * @param p Pointer to LinearParameters to initialize
 * 
 * @note Values are uniformly distributed in [0, 1]
 */
void linear_random_init(struct LinearParameters *p);

/**
 * @brief Free memory associated with linear layer parameters
 * 
 * @param p Pointer to LinearParameters to free
 */
void linear_free(struct LinearParameters* p);

/**
 * @brief Perform forward pass through linear layer
 * 
 * Computes: y = xW^T + b with bias broadcasting
 * 
 * @param x Pointer to input matrix [batch_size x in_features]
 * @param p Pointer to layer parameters
 * @return Output matrix [batch_size x out_features]
 * 
 * @note The returned matrix must be freed by the caller
 */
struct Matrix2D linear_forward(const struct Matrix2D *x, const struct LinearParameters *p);

/* ========================================
 * Self-Attention Layer Functions
 * ========================================
 * Self-attention computes attention scores between all positions in a sequence
 */

/**
 * @brief Create a new self-attention layer with uninitialized parameters
 * 
 * @param embeded_dim Embedding dimension (same for input and output)
 * @return SelfAttentionParameters structure with allocated Q, K, V, O projections
 * 
 * @note All four projection matrices (Wq, Wk, Wv, Wo) will be [embed_dim x embed_dim]
 * @note Call attention_random_init() to initialize with random values
 * @note Call attention_free() when done to release memory
 */
struct SelfAttentionParameters attention_new(unsigned int embeded_dim);

/**
 * @brief Create a deep copy of self-attention layer parameters
 * 
 * @param p Pointer to source SelfAttentionParameters to copy
 * @return New SelfAttentionParameters with all projections copied
 * 
 * @note The returned structure must be freed with attention_free()
 */
struct SelfAttentionParameters attention_copy(const struct SelfAttentionParameters *p);

/**
 * @brief Initialize self-attention parameters with random values
 * 
 * @param p Pointer to SelfAttentionParameters to initialize
 */
void attention_random_init(struct SelfAttentionParameters *p);

/**
 * @brief Free memory associated with self-attention layer
 * 
 * @param p Pointer to SelfAttentionParameters to free
 */
void attention_free(struct SelfAttentionParameters* p);

/**
 * @brief Perform forward pass through self-attention layer with residual connection
 * 
 * Algorithm:
 * 1. Project to Q, K, V: Q = xWq, K = xWk, V = xWv
 * 2. Compute scaled attention scores: scores = (QK^T) / sqrt(d_k)
 * 3. Apply causal mask (prevent attending to future positions)
 * 4. Apply softmax to get attention weights
 * 5. Compute weighted values: attention_out = weights * V
 * 6. Apply output projection: out = attention_out * Wo
 * 7. Add residual connection: result = x + out
 * 
 * @param x Pointer to input matrix [seq_len x embed_dim]
 * @param p Pointer to layer parameters
 * @return Output matrix [seq_len x embed_dim] with residual connection applied
 * 
 * @note The returned matrix must be freed by the caller
 * @note Includes residual connection (output += input)
 */
struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameters *p);

/* ========================================
 * Embeddings Layer Functions
 * ========================================
 * Embeddings convert discrete token IDs into continuous vector representations
 */

/**
 * @brief Create a new embeddings layer with uninitialized parameters
 * 
 * @param vocab_size Size of vocabulary (number of unique tokens)
 * @param embeded_dim Dimension of embedding vectors
 * @return EmbeddingsParameters structure with allocated weight matrix
 * 
 * @note Weight matrix will be [vocab_size x embed_dim]
 * @note Call embeddings_random_init() to initialize with random values
 * @note Call embeddings_free() when done to release memory
 */
struct EmbeddingsParameters embeddings_new(unsigned int vocab_size, unsigned int embeded_dim);

/**
 * @brief Create a deep copy of embeddings layer parameters
 * 
 * @param p Pointer to source EmbeddingsParameters to copy
 * @return New EmbeddingsParameters with copied weight matrix
 * 
 * @note The returned structure must be freed with embeddings_free()
 */
struct EmbeddingsParameters embeddings_copy(const struct EmbeddingsParameters *p);

/**
 * @brief Initialize embeddings with random values
 * 
 * @param p Pointer to EmbeddingsParameters to initialize
 */
void embeddings_random_init(struct EmbeddingsParameters *p);

/**
 * @brief Free memory associated with embeddings layer
 * 
 * @param p Pointer to EmbeddingsParameters to free
 */
void embeddings_free(struct EmbeddingsParameters* p);

/**
 * @brief Look up embeddings for given token indices
 * 
 * Performs a table lookup: for each index i in indices, retrieves row i
 * from the embedding weight matrix.
 * 
 * @param indices Pointer to matrix of token indices [1 x seq_len]
 * @param p Pointer to layer parameters
 * @return Matrix of embedding vectors [seq_len x embed_dim]
 * 
 * @note The returned matrix must be freed by the caller
 * @note Each index must be < vocab_size
 */
struct Matrix2D embeddings_forward(const struct Matrix2D_UInt *indices, const struct EmbeddingsParameters *p);


#endif // _RLM_LAYERS_H