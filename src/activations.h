#ifndef _RLM_ACTIVATIONS_H
#define _RLM_ACTIVATIONS_H

struct Matrix2D relu(const struct Matrix2D *m);
struct Matrix2D softmax(const struct Matrix2D *m);

#endif // _RLM_ACTIVATIONS_H