#ifndef _RLM_TRANSFORMER_H
#define _RLM_TRANSFORMER_H

#include "layers.h"
#include "matrix.h"

struct TransformerParameters {
	/* Embeddings */
	struct EmbeddingsParameters token_embed;
	struct Matrix2D pos_embed;

	/* Attention linear projections */
	struct SelfAttentionParameters attn;

	/* Feed-forward */
	struct LinearParameters W1;
	struct LinearParameters W2;

	/* LM head */
	struct LinearParameters lm_head;
};

/* ========================================
 * Transformer Model Functions
 * ======================================== */

struct TransformerParameters transformer_new(unsigned int seq_len, unsigned int embeded_dim, unsigned int ff_dim, unsigned int vocab_size);

void transformer_free(struct TransformerParameters* p);

struct Matrix2D transformer_forward(const struct Matrix2D_UInt *x, 
                                           const struct TransformerParameters *p,
                                           struct AttentionCache *cache,
                                           unsigned int pos);

#endif // _RLM_TRANSFORMER_H