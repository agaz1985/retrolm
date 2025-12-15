#ifndef _RLM_TRANSFORMER_H
#define _RLM_TRANSFORMER_H

#include "layers.h"
#include "matrix.h"

struct TransformerParameters {
	// TODO AGAZ: self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM) * 0.1)

	/* Attention linear projections */
	struct SelfAttentionParameter attn;

	/* Feed-forward */
	struct LinearParameters W1; // = nn.Linear(EMBED_DIM, FF_DIM)
	struct LinearParameters W2; // = nn.Linear(FF_DIM, EMBED_DIM)

	/* LM head */
	struct LinearParameters lm_head; // = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
};

/* Linear layer with bias */

struct TransformerParameters transformer_new(unsigned int in_features, unsigned int out_features);
void transformer_free(struct LinearParameters* p);
struct Matrix2D transformer_forward(const struct Matrix2D *x, const struct LinearParameters *p);


#endif // _RLM_TRANSFORMER_H