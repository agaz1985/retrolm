#include "activations.h"
#include "matrix.h"

struct Matrix2D relu(const struct Matrix2D *m) {
	return mat_clamp_min(m, 0.0);
}

struct Matrix2D softmax(const struct Matrix2D *m) {
	struct Matrix2D num = mat_exp(m);
	struct Matrix2D den = mat_sum(&num, 1);
	struct Matrix2D result = mat_div(&num, &den);

	mat_free(&num);
	mat_free(&den);

	return result;
}