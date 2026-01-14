#include "transformer.h"
#include "activations.h"
#include "layers.h"
#include "exceptions.h"
#include <stdio.h>

struct TransformerParameters transformer_new(unsigned int seq_len, unsigned int embeded_dim, unsigned int ff_dim, unsigned int vocab_size) {
	struct TransformerParameters params;
	params.token_embed = embeddings_new(vocab_size, embeded_dim);
	params.pos_embed = mat_new(seq_len, embeded_dim);
	params.attn = attention_new(embeded_dim);
	params.W1 = linear_new(embeded_dim, ff_dim);
	params.W2 = linear_new(ff_dim, embeded_dim);
	params.lm_head = linear_new(embeded_dim, vocab_size);
	return params;
}

void transformer_free(struct TransformerParameters* p) {
	embeddings_free(&p->token_embed);
	mat_free(&p->pos_embed);
	attention_free(&p->attn);
	linear_free(&p->W1);
	linear_free(&p->W2);
	linear_free(&p->lm_head);
}

struct Matrix2D transformer_forward(const struct Matrix2D_UInt *x,
                                    const struct TransformerParameters *p,
                                    struct AttentionCache *cache,
                                    unsigned int pos) {
	if (x->r > 1) {
		throw("Batch processing not supported.\n", InvalidInput);
	}
	const unsigned int N = x->c;  // Number of tokens in current input
	
	/* Token embeddings */
	struct Matrix2D X = embeddings_forward(x, &p->token_embed);
	
	/* Positional embeddings - use current position */
	struct Matrix2D_UInt posEmbedIndices = mat_uint_new(1, N);
	for (unsigned int i = 0; i < N; i++) {
		unsigned int idx = pos + i;
		// Check bounds to prevent accessing beyond position embeddings
		if (idx >= p->pos_embed.r) {
			char msg[256];
			snprintf(msg, sizeof(msg), "Position index %u exceeds maximum sequence length %u\n", idx, p->pos_embed.r);
			mat_uint_free(&posEmbedIndices);
			mat_free(&X);
			throw(msg, InvalidInput);
		}
		posEmbedIndices.data[i] = idx;
	}
	struct Matrix2D pS = mat_rowselect(&p->pos_embed, &posEmbedIndices);
	struct Matrix2D X_new = mat_add(&X, &pS);
	mat_free(&X);
	mat_uint_free(&posEmbedIndices);
	mat_free(&pS);
	X = X_new;
	
	/* Self-Attention with provided cache */
	X_new = attention_forward(&X, &p->attn, cache);
	mat_free(&X);
	X = X_new;
	
	/* Feed-Forward with Residual */
	struct Matrix2D X_residual = mat_copy(&X);
	struct Matrix2D FF = linear_forward(&X, &p->W1);
	struct Matrix2D FF_relu = relu(&FF);
	mat_free(&FF);
	FF = linear_forward(&FF_relu, &p->W2);
	mat_free(&FF_relu);
	X_new = mat_add(&X_residual, &FF);
	mat_free(&X);
	mat_free(&X_residual);
	mat_free(&FF);
	X = X_new;
	
	/* LM head */
	struct Matrix2D logits = linear_forward(&X, &p->lm_head);
	
	/* Deallocate memory */
	mat_free(&X);
	
	return logits;
}
