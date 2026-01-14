#include <math.h>
#include <stddef.h>
#include "activations.h"
#include "exceptions.h"
#include "layers.h"

/* ========================================
 * Linear Layer Implementation
 * ======================================== */

struct LinearParameters linear_new(unsigned int in_features, unsigned int out_features) {
	struct LinearParameters params;
	params.weights = mat_new(out_features, in_features);
	params.bias = mat_new(1, out_features);
	return params;
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

/* ========================================
 * Self-Attention Layer Implementation
 * ======================================== */

/* Forward declaration */
static struct Matrix2D attention_forward_single(const struct Matrix2D *x, const struct Matrix2D *Q, const struct Matrix2D *K, const struct Matrix2D *V, const struct SelfAttentionParameters *p);

struct SelfAttentionParameters attention_new(unsigned int embed_dim) {
	struct SelfAttentionParameters params;
	params.Wq = linear_new(embed_dim, embed_dim);
	params.Wk = linear_new(embed_dim, embed_dim);
	params.Wv = linear_new(embed_dim, embed_dim);
	params.Wo = linear_new(embed_dim, embed_dim);
	return params;
}

void attention_free(struct SelfAttentionParameters* p) {
	linear_free(&p->Wq);
	linear_free(&p->Wk);
	linear_free(&p->Wv);
	linear_free(&p->Wo);
}

struct AttentionCache attention_cache_new(unsigned int embed_dim) {
	struct AttentionCache cache;
	cache.K = mat_new(0, embed_dim);
	cache.V = mat_new(0, embed_dim);
	return cache;
}

void attention_cache_free(struct AttentionCache *cache) {
	mat_free(&cache->K);
	mat_free(&cache->V);
}

static struct Matrix2D mat_vstack(const struct Matrix2D *m1, const struct Matrix2D *m2) {
	if (m1->c != m2->c) {
		throw("Cannot vstack matrices with different column counts", InvalidInput);
	}
	
	const unsigned int total_rows = m1->r + m2->r;
	if (total_rows == 0) {
		return mat_new(0, m1->c);
	}
	
	struct Matrix2D result = mat_new(total_rows, m1->c);
	
	// Copy first matrix rows
	for (unsigned int i = 0; i < m1->r; i++) {
		for (unsigned int j = 0; j < m1->c; j++) {
			result.data[i * result.c + j] = m1->data[i * m1->c + j];
		}
	}
	
	// Copy second matrix rows
	for (unsigned int i = 0; i < m2->r; i++) {
		for (unsigned int j = 0; j < m2->c; j++) {
			result.data[(m1->r + i) * result.c + j] = m2->data[i * m2->c + j];
		}
	}
	
	return result;
}

struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameters *p, struct AttentionCache *cache) {
	// Compute Q, K, V for the input tokens (all rows of x processed in one matrix operation)
	// If x is [n_tokens x embed_dim], then Q, K_new, V_new are also [n_tokens x embed_dim]
	struct Matrix2D Q = linear_forward(x, &p->Wq);
	struct Matrix2D K_new = linear_forward(x, &p->Wk);
	struct Matrix2D V_new = linear_forward(x, &p->Wv);
	
	struct Matrix2D K_full;
	struct Matrix2D V_full;
	
	if (cache->K.r > 0) {
		// Concatenate cached K,V (previous tokens) with new K,V (current tokens)
		// K_full and V_full will have shape [total_tokens x embed_dim]
		// where total_tokens = cached_tokens + n_tokens
		K_full = mat_vstack(&cache->K, &K_new);
		V_full = mat_vstack(&cache->V, &V_new);
		
		mat_free(&K_new);
		mat_free(&V_new);
	} else {
		// First call: no cached tokens yet, use new K,V directly
		K_full = K_new;
		V_full = V_new;
	}
	
	// Update cache with full K,V for next iteration
	mat_free(&cache->K);
	mat_free(&cache->V);
	cache->K = mat_copy(&K_full);
	cache->V = mat_copy(&V_full);
	
	// Run attention computation: Q attends to all tokens in K_full, V_full
	struct Matrix2D result = attention_forward_single(x, &Q, &K_full, &V_full, p);
	
	// Clean up
	mat_free(&Q);
	mat_free(&K_full);
	mat_free(&V_full);
	
	return result;
}

struct Matrix2D attention_forward_single(const struct Matrix2D *x, const struct Matrix2D *Q, const struct Matrix2D *K, const struct Matrix2D *V, const struct SelfAttentionParameters *p) {
	// Compute attention scores: Q * K^T
	// Result shape: [n_tokens x total_tokens] where result[i,j] = similarity of query_i to key_j
	struct Matrix2D K_t = mat_transpose(K);
	struct Matrix2D scores = mat_mul(Q, &K_t);
	mat_free(&K_t);
	
	// Scale by sqrt(d_k)
	const unsigned int embed_dim = p->Wq.weights.r;
	mat_scale(&scores, 1.0 / sqrt(embed_dim));

	// Causal mask
	mat_maskdiag(&scores, -HUGE_VALF);

	// Apply softmax
	struct Matrix2D weights = softmax(&scores);
	mat_free(&scores);
	
	// Compute attention output: weights * V
	struct Matrix2D attention_out = mat_mul(&weights, V);
	mat_free(&weights);
	
	// Apply output projection
	struct Matrix2D projected = linear_forward(&attention_out, &p->Wo);
	mat_free(&attention_out);
	
	// Add residual connection
	struct Matrix2D result = mat_add(x, &projected);
	mat_free(&projected);
	
	return result;
}

/* ========================================
 * Embeddings Layer Implementation
 * ======================================== */

struct EmbeddingsParameters embeddings_new(unsigned int vocab_size, unsigned int embed_dim) {
	struct EmbeddingsParameters params;
	params.weight_matrix = mat_new(vocab_size, embed_dim);
	return params;
}

void embeddings_free(struct EmbeddingsParameters* p) {
	mat_free(&p->weight_matrix);
}

struct Matrix2D embeddings_forward(const struct Matrix2D_UInt *indices, const struct EmbeddingsParameters *p) {
	struct Matrix2D embedding_vectors = mat_rowselect(&p->weight_matrix, indices);
	return embedding_vectors;
}