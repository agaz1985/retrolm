#ifndef _RLM_LOADER_H
#define _RLM_LOADER_H

#include "matrix.h"
#include "transformer.h"

struct Matrix2D load_weight_matrix(const char *filepath);
struct TransformerParameters load_model_weights(const char *weights_dir);

#endif // _RLM_LOADER_H