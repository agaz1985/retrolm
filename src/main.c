#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "matrix.h"
#include "activations.h"
#include "layers.h"

int main() {
	logger("Allocate memory...\n", DEBUG);

	struct Matrix2D m1 = mat_new(2, 3);
	struct Matrix2D m2 = mat_new(3, 2);
	struct Matrix2D m4 = mat_new(1, 3);

	/* m1: 2 rows, 3 cols */
	*mat_at(&m1, 0, 0) = 1.f;
	*mat_at(&m1, 0, 1) = 2.f;
	*mat_at(&m1, 0, 2) = 3.f;
	*mat_at(&m1, 1, 0) = 4.f;
	*mat_at(&m1, 1, 1) = 5.f;
	*mat_at(&m1, 1, 2) = 6.f;

	/* m2: 3 rows, 2 cols */
	*mat_at(&m2, 0, 0) = 1.f;
	*mat_at(&m2, 0, 1) = 2.f;
	*mat_at(&m2, 1, 0) = 3.f;
	*mat_at(&m2, 1, 1) = 4.f;
	*mat_at(&m2, 2, 0) = 5.f;
	*mat_at(&m2, 2, 1) = 6.f;

	/* m2: 1 rows, 3 cols */
	*mat_at(&m4, 0, 0) = 1.f;
	*mat_at(&m4, 0, 1) = -2.f;
	*mat_at(&m4, 0, 2) = 3.f;

	logger("Matrix operations...\n", INFO);
	struct Matrix2D m3 = mat_mul(&m1, &m2);
	logger("Multiply:\n", DEBUG);
	mat_print(&m3);

	mat_scale(&m3, 2.0);
	logger("Scale:\n", DEBUG);
	mat_print(&m3);

	mat_shift(&m3, 10.0);
	logger("Shift:\n", DEBUG);
	mat_print(&m3);

	struct Matrix2D m3_t = mat_transpose(&m3);
	struct Matrix2D m_i = mat_identity(5);

	logger("Transpose:\n", DEBUG);
	mat_print(&m3_t);
	logger("Identity:\n", DEBUG);
	mat_print(&m_i);

	struct Matrix2D m5 = relu(&m4);
	logger("Relu:\n", DEBUG);
	mat_print(&m5);

	// Try linear layer.
	struct LinearParameters linear_params = linear_new(2, 5);
	linear_random_init(&linear_params);

	struct Matrix2D y = linear_forward(&m2, &linear_params);
	logger("Linear:\n", DEBUG);
	mat_print(&y);

	logger("Deallocate memory...\n", DEBUG);
	mat_free(&m1);
	mat_free(&m2);
	mat_free(&m3);
	mat_free(&m4);
	mat_free(&m5);
	mat_free(&m3_t);
	mat_free(&m_i);
	linear_free(&linear_params);
	mat_free(&y);

	return 0;
}