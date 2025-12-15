#ifndef _RLM_LAYERS_H
#define _RLM_LAYERS_H

#include "matrix.h"

struct LinearParameters {
	struct Matrix2D weights;
	struct Matrix2D bias;
};

struct SelfAttentionParameter {
	struct LinearParameters Wq;
	struct LinearParameters Wk;
	struct LinearParameters Wv;
	struct LinearParameters Wo;	
};

/* Linear layer with bias */

struct LinearParameters linear_new(unsigned int in_features, unsigned int out_features);
struct LinearParameters linear_copy(const struct LinearParameters *p);
void linear_random_init(struct LinearParameters *p);
void linear_free(struct LinearParameters* p);
struct Matrix2D linear_forward(const struct Matrix2D *x, const struct LinearParameters *p);

/* Attention layer */

struct SelfAttentionParameter attention_new(unsigned int embeded_dim);
struct SelfAttentionParameter attention_copy(const struct SelfAttentionParameter *p);
void attention_random_init(struct SelfAttentionParameter *p);
void attention_free(struct SelfAttentionParameter* p);
struct Matrix2D attention_forward(const struct Matrix2D *x, const struct SelfAttentionParameter *p);

#endif // _RLM_LAYERS_H