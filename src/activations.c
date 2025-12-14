#include "activations.h"
#include "matrix.h"

struct Matrix2D relu(const struct Matrix2D *m) {
	return mat_clamp_min(m, 0.0);
}