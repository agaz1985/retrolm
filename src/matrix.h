#ifndef _RLM_MATRIX_H
#define _RLM_MATRIX_H

struct Matrix2D {
	unsigned int r;
	unsigned int c;
	float* data;
};

struct Matrix2D_UInt {
	unsigned int r;
	unsigned int c;
	unsigned int* data;
};

/* ========================================
 * Construction and Destruction
 * ======================================== */

struct Matrix2D mat_new(unsigned int r, unsigned int c);

void mat_free(struct Matrix2D *m);

struct Matrix2D_UInt mat_uint_new(unsigned int r, unsigned int c);

void mat_uint_free(struct Matrix2D_UInt *m);

struct Matrix2D_UInt indices_new(unsigned int n);

/* ========================================
 * Element Access and Manipulation
 * ======================================== */

float *mat_at(const struct Matrix2D *m,
              unsigned int i,
              unsigned int j);

void mat_maskdiag(struct Matrix2D *m, float value);

/* ========================================
 * Utilities
 * ======================================== */

void mat_print(const struct Matrix2D *m);

/* ========================================
 * Core Linear Algebra Operations
 * ======================================== */

struct Matrix2D mat_mul(const struct Matrix2D *m1, const struct Matrix2D *m2);

struct Matrix2D mat_div(const struct Matrix2D *m1, const struct Matrix2D *m2);

struct Matrix2D mat_add(const struct Matrix2D *m1, const struct Matrix2D *m2);

struct Matrix2D mat_sub(const struct Matrix2D *m1, const struct Matrix2D *m2);

struct Matrix2D mat_exp(const struct Matrix2D *m);

struct Matrix2D mat_sum(const struct Matrix2D *m, unsigned short dim);

struct Matrix2D mat_max(const struct Matrix2D *m, unsigned short dim);

/* ========================================
 * In-Place Scalar Operations
 * ======================================== */

void mat_scale(struct Matrix2D *m, float alpha);

/* ========================================
 * Matrix Transformations
 * ======================================== */

struct Matrix2D mat_transpose(const struct Matrix2D *m);

/* ========================================
 * Copy and Clamping Operations
 * ======================================== */

struct Matrix2D mat_copy(const struct Matrix2D *m);

struct Matrix2D mat_clamp_min(const struct Matrix2D *m, float lo);

struct Matrix2D mat_rowselect(const struct Matrix2D *m, const struct Matrix2D_UInt *indices);

#endif // _RLM_MATRIX_H