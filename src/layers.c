#include <math.h>

#include "activations.h"
#include "layers.h"

/* Linear layer with bias */

struct LinearParameters linear_new(unsigned int in_features, unsigned int out_features) {
	struct LinearParameters params;
	params.weights = mat_new(out_features, in_features);
	params.bias = mat_new(1, out_features);
	return params;
}

struct LinearParameters linear_copy(const struct LinearParameters *p) {
	struct LinearParameters params;
	params.weights = mat_copy(&p->weights);
	params.bias = mat_copy(&p->bias);
	return params;
}

void linear_random_init(struct LinearParameters *p) {
	mat_random_init(&p->weights);
	mat_random_init(&p->bias);
}

void linear_free(struct LinearParameters* p) {
	mat_free(&p->weights);
	mat_free(&p->bias);
}

struct Matrix2D linear_forward(const struct Matrix2D *x, const struct LinearParameters *p) {
	/* y = xA_t + b */
	const struct Matrix2D weights_t = mat_transpose(&p->weights);
	const struct Matrix2D product = mat_mul(x, &weights_t);
	return mat_add(&product, &p->bias);
}

/* Attention layer */

struct SelfAttentionParameters attention_new(unsigned int embeded_dim) {
	struct SelfAttentionParameters params;
	params.Wq = linear_new(embeded_dim, embeded_dim);
	params.Wk = linear_new(embeded_dim, embeded_dim);
	params.Wv = linear_new(embeded_dim, embeded_dim);
	params.Wo = linear_new(embeded_dim, embeded_dim);
	return params;
}

struct SelfAttentionParameters attention_copy(const struct SelfAttentionParameters *p) {
	struct SelfAttentionParameters params;
	params.Wq = linear_copy(&p->Wq);
	params.Wk = linear_copy(&p->Wk);
	params.Wv = linear_copy(&p->Wv);
	params.Wo = linear_copy(&p->Wo);
	return params;
}

void attention_random_init(struct SelfAttentionParameters *p) {
	linear_random_init(&p->Wq);
	linear_random_init(&p->Wk);
	linear_random_init(&p->Wv);
	linear_random_init(&p->Wo);
}

void attention_free(struct SelfAttentionParameters* p) {
	linear_free(&p->Wq);
	linear_free(&p->Wk);
	linear_free(&p->Wv);
	linear_free(&p->Wo);
}

struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameters *p) {
	const struct Matrix2D Q = linear_forward(x, &p->Wq);
	const struct Matrix2D K = linear_forward(x, &p->Wk);
	const struct Matrix2D V = linear_forward(x, &p->Wv);

	const struct Matrix2D K_t = mat_transpose(&K);
	struct Matrix2D scores = mat_mul(&Q, &K_t);

	const unsigned int embded_dim = p->Wq.weights.r;
	mat_scale(&scores, 1.0 / sqrt(embded_dim));

	const struct Matrix2D weights = softmax(&scores);
	const struct Matrix2D attention_out = mat_mul(&weights, &V);
	const struct Matrix2D residual = linear_forward(&attention_out, &p->Wo);
	return mat_add(x, &residual);
}

struct EmbeddingsParameters embeddings_new(unsigned int vocab_size, unsigned int embeded_dim) {
	struct EmbeddingsParameters params;
	params.weight_matrix = mat_new(vocab_size, embeded_dim);
	return params;
}

struct EmbeddingsParameters embeddings_copy(const struct EmbeddingsParameters *p) {
	struct EmbeddingsParameters params;
	params.weight_matrix = mat_copy(&p->weight_matrix);
	return params;
}

void embeddings_random_init(struct EmbeddingsParameters *p) {
	mat_random_init(&p->weight_matrix);
}

void embeddings_free(struct EmbeddingsParameters* p) {
	mat_free(&p->weight_matrix);
}

struct Matrix2D embeddings_forward(const unsigned int *indices, unsigned int n_indices, const struct EmbeddingsParameters *p) {
	struct Matrix2D embedding_vectors = mat_rowselect(&p->weight_matrix, indices, n_indices);
	return embedding_vectors;
}
