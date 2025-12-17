#include "transformer.h"
#include "activations.h"
#include "layers.h"
#include "exceptions.h"

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

struct TransformerParameters transformer_copy(const struct TransformerParameters *p) {
	struct TransformerParameters params;
	params.token_embed = embeddings_copy(&p->token_embed);
	params.pos_embed = mat_copy(&p->pos_embed);
	params.attn = attention_copy(&p->attn);
	params.W1 = linear_copy(&p->W1);
	params.W2 = linear_copy(&p->W2);
	params.lm_head = linear_copy(&p->lm_head);
	return params;
}

void transformer_random_init(struct TransformerParameters *p) {
	embeddings_random_init(&p->token_embed);
	mat_random_init(&p->pos_embed);
	attention_random_init(&p->attn);
	linear_random_init(&p->W1);
	linear_random_init(&p->W2);
	linear_random_init(&p->lm_head);
}

void transformer_free(struct TransformerParameters* p) {
	embeddings_free(&p->token_embed);
	mat_free(&p->pos_embed);
	attention_free(&p->attn);
	linear_free(&p->W1);
	linear_free(&p->W2);
	linear_free(&p->lm_head);
}

struct Matrix2D transformer_forward(const struct Matrix2D_UInt *x, const struct TransformerParameters *p) {
	if (x->r > 1) {
		throw("Batch processing not supported.\n", InvalidInput);
	}
	const unsigned int N = x->c; // [BxN] a.k.a [1xN]

	struct Matrix2D X = embeddings_forward(x, &p->token_embed);

	struct Matrix2D_UInt posEmbedIndices = indices_new(N);
	struct Matrix2D pS = mat_rowselect(&p->pos_embed, &posEmbedIndices);
    X = mat_add(&X, &pS);  // [seq, embed]

    /* Self-Attention */
    X = attention_forward(&X, &p->attn);

    /* Feed-Forward */
    X = linear_forward(&X, &p->W1);
    struct Matrix2D FF = relu(&X);

    FF = linear_forward(&FF, &p->W2);
    X = mat_add(&X, &FF); // residual

    /* LM head */
    struct Matrix2D logits = linear_forward(&X, &p->lm_head);

    /* Deallocate memory*/
    mat_free(&X);
    mat_uint_free(&posEmbedIndices);
    mat_free(&pS);
    mat_free(&FF);

	return logits;
}
