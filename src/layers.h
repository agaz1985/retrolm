#ifndef _RLM_LAYERS_H
#define _RLM_LAYERS_H

#include "matrix.h"

struct LinearParameters {
	struct Matrix2D weights;
	struct Matrix2D bias;
};

struct AttentionCache {
	struct Matrix2D K;
	struct Matrix2D V;
};

struct SelfAttentionParameters {
	struct LinearParameters Wq;
	struct LinearParameters Wk;
	struct LinearParameters Wv;
	struct LinearParameters Wo;
};

struct EmbeddingsParameters {
	struct Matrix2D weight_matrix;
};

struct LinearParameters linear_new(unsigned int in_features, unsigned int out_features);

void linear_free(struct LinearParameters* p);

struct Matrix2D linear_forward(const struct Matrix2D *x, const struct LinearParameters *p);

struct SelfAttentionParameters attention_new(unsigned int embeded_dim);

void attention_free(struct SelfAttentionParameters* p);

struct AttentionCache attention_cache_new(unsigned int embed_dim);

void attention_cache_free(struct AttentionCache *cache);

struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameters *p, struct AttentionCache *cache);

struct EmbeddingsParameters embeddings_new(unsigned int vocab_size, unsigned int embeded_dim);


void embeddings_free(struct EmbeddingsParameters* p);

struct Matrix2D embeddings_forward(const struct Matrix2D_UInt *indices, const struct EmbeddingsParameters *p);


#endif // _RLM_LAYERS_H