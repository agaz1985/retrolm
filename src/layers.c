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

struct SelfAttentionParameter attention_new(unsigned int embeded_dim) {
	struct SelfAttentionParameter params;
	params.Wq = mat_new(embeded_dim, embeded_dim);
	params.Wk = mat_new(embeded_dim, embeded_dim);
	params.Wv = mat_new(embeded_dim, embeded_dim);
	params.Wo = mat_new(embeded_dim, embeded_dim);
	return params;
}

struct SelfAttentionParameter attention_copy(const struct SelfAttentionParameter *p) {
	struct SelfAttentionParameter params;
	params.Wq = mat_copy(&p->Wq);
	params.Wk = mat_copy(&p->Wk);
	params.Wv = mat_copy(&p->Wv);
	params.Wo = mat_copy(&p->Wo);
	return params;
}

void attention_random_init(struct SelfAttentionParameter *p) {
	mat_random_init(&p->Wq);
	mat_random_init(&p->Wk);
	mat_random_init(&p->Wv);
	mat_random_init(&p->Wo);
}

void attention_free(struct SelfAttentionParameter* p) {
	mat_free(&p->Wq);
	mat_free(&p->Wk);
	mat_free(&p->Wv);
	mat_free(&p->Wo);
}

struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameter *p) {
	const struct Matrix2D Q = mat_mul(x, &p->Wq);
	const struct Matrix2D K = mat_mul(x, &p->Wk);
	const struct Matrix2D V = mat_mul(x, &p->Wv);

	const struct Matrix2D K_t = mat_transpose(&K);
	struct Matrix2D scores = mat_mul(&Q, &K_t);

	const unsigned int embded_dim = p->Wq.r;
	mat_scale(&scores, 1.0 / sqrt(embded_dim));

	const struct Matrix2D weights = softmax(&scores);
	const struct Matrix2D attention_out = mat_mul(&weights, &V);
	const struct Matrix2D residual = mat_mul(&Wo, &attention_out);
	return mat_add(&x, &residual);
}
