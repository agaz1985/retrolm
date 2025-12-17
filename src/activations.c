#include "activations.h"
#include "matrix.h"

struct Matrix2D relu(const struct Matrix2D *m) {
	return mat_clamp_min(m, 0.0);
}

struct Matrix2D softmax(const struct Matrix2D *m) {
    struct Matrix2D max_vals = mat_max(m, 1);
    struct Matrix2D m_shifted = mat_sub(m, &max_vals);
    struct Matrix2D num = mat_exp(&m_shifted);    
    struct Matrix2D den = mat_sum(&num, 1);
    struct Matrix2D result = mat_div(&num, &den);

    // Cleanup
    mat_free(&max_vals);
    mat_free(&m_shifted);
    mat_free(&num);
    mat_free(&den);
    
    return result;
}