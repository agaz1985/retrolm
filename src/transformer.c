#include "transformer.h"

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

struct Matrix2D transformer_forward(const struct Matrix2D *x, const struct TransformerParameters *p) {
	// TODO AGAZ: implement.
	struct Matrix2D res;
	return res;
}
