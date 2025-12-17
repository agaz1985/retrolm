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
	struct Matrix2D weights_t = mat_transpose(&p->weights);
	struct Matrix2D product = mat_mul(x, &weights_t);
	struct Matrix2D result = mat_add(&product, &p->bias);
	
	mat_free(&weights_t);
	mat_free(&product);
	
	return result;
}

/* Attention layer */
struct SelfAttentionParameters attention_new(unsigned int embed_dim) {
	struct SelfAttentionParameters params;
	params.Wq = linear_new(embed_dim, embed_dim);
	params.Wk = linear_new(embed_dim, embed_dim);
	params.Wv = linear_new(embed_dim, embed_dim);
	params.Wo = linear_new(embed_dim, embed_dim);
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
	// Compute Q, K, V
	struct Matrix2D Q = linear_forward(x, &p->Wq);
	struct Matrix2D K = linear_forward(x, &p->Wk);
	struct Matrix2D V = linear_forward(x, &p->Wv);
	
	// Compute attention scores: Q * K^T
	struct Matrix2D K_t = mat_transpose(&K);
	struct Matrix2D scores = mat_mul(&Q, &K_t);
	mat_free(&Q);
	mat_free(&K);
	mat_free(&K_t);
	
	// Scale by sqrt(d_k)
	const unsigned int embed_dim = p->Wq.weights.r;
	mat_scale(&scores, 1.0 / sqrt(embed_dim));
	
	// Apply softmax
	struct Matrix2D weights = softmax(&scores);
	mat_free(&scores);
	
	// Compute attention output: weights * V
	struct Matrix2D attention_out = mat_mul(&weights, &V);
	mat_free(&weights);
	mat_free(&V);
	
	// Apply output projection
	struct Matrix2D residual = linear_forward(&attention_out, &p->Wo);
	mat_free(&attention_out);
	
	// Add residual connection
	struct Matrix2D result = mat_add(x, &residual);
	mat_free(&residual);
	
	return result;
}

/* Embeddings layer */
struct EmbeddingsParameters embeddings_new(unsigned int vocab_size, unsigned int embed_dim) {
	struct EmbeddingsParameters params;
	params.weight_matrix = mat_new(vocab_size, embed_dim);
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

struct Matrix2D embeddings_forward(const struct Matrix2D_UInt *indices, const struct EmbeddingsParameters *p) {
	struct Matrix2D embedding_vectors = mat_rowselect(&p->weight_matrix, indices);
	return embedding_vectors;
}