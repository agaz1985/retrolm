/**
 * @file transformer.h
 * @brief Transformer language model architecture
 * 
 * Implements a simplified transformer decoder for language modeling with:
 * - Token and positional embeddings
 * - Single-layer self-attention
 * - Feed-forward network with residual connections
 * - Language modeling head
 */

#ifndef _RLM_TRANSFORMER_H
#define _RLM_TRANSFORMER_H

#include "layers.h"
#include "matrix.h"

/**
 * @brief Complete transformer model parameters
 * 
 * Contains all learnable parameters for a single-layer transformer decoder.
 * Weight tying is used between token embeddings and LM head.
 */
struct TransformerParameters {
	/* Embeddings */
	struct EmbeddingsParameters token_embed;  /**< Token embeddings [vocab_size x embed_dim] */
	struct Matrix2D pos_embed;                /**< Positional embeddings [seq_len x embed_dim] */

	/* Attention linear projections */
	struct SelfAttentionParameters attn;      /**< Self-attention layer */

	/* Feed-forward */
	struct LinearParameters W1;               /**< First FF layer [embed_dim x ff_dim] */
	struct LinearParameters W2;               /**< Second FF layer [ff_dim x embed_dim] */

	/* LM head */
	struct LinearParameters lm_head;          /**< Output projection [embed_dim x vocab_size] */
};

/* ========================================
 * Transformer Model Functions
 * ======================================== */

/**
 * @brief Create a new transformer model with uninitialized parameters
 * 
 * @param seq_len Maximum sequence length
 * @param embeded_dim Embedding dimension
 * @param ff_dim Feed-forward hidden dimension
 * @param vocab_size Vocabulary size
 * @return TransformerParameters structure with allocated layers
 * 
 * @note Call transformer_random_init() to initialize with random weights
 * @note Call transformer_free() when done
 */
struct TransformerParameters transformer_new(unsigned int seq_len, unsigned int embeded_dim, unsigned int ff_dim, unsigned int vocab_size);

/**
 * @brief Create a deep copy of transformer parameters
 * 
 * @param p Pointer to transformer parameters to copy
 * @return New TransformerParameters with all weights copied
 */
struct TransformerParameters transformer_copy(const struct TransformerParameters *p);

/**
 * @brief Initialize all transformer parameters with random values
 * 
 * @param p Pointer to TransformerParameters to initialize
 */
void transformer_random_init(struct TransformerParameters *p);

/**
 * @brief Free all transformer memory
 * 
 * @param p Pointer to TransformerParameters to free
 */
void transformer_free(struct TransformerParameters* p);

/**
 * @brief Forward pass through transformer model
 * 
 * Architecture:
 * 1. Token embeddings + positional embeddings
 * 2. Self-attention with residual connection
 * 3. Feed-forward network with residual connection
 * 4. Language modeling head projection
 * 
 * @param x Token indices [1 x seq_len]
 * @param p Pointer to model parameters
 * @return Logits matrix [seq_len x vocab_size]
 * 
 * @throws InvalidInput if x has more than one row (batching not supported)
 * @note Returned matrix must be freed by caller
 */
struct Matrix2D transformer_forward(const struct Matrix2D_UInt *x, const struct TransformerParameters *p);

#endif // _RLM_TRANSFORMER_H