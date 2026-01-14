#ifndef _RLM_MATRIX_OPS_H
#define _RLM_MATRIX_OPS_H

void _matmul(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1, unsigned int c2);

/* ========================================
 * Element-wise addition operations
 * ======================================== */

void _matadd(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matadd_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matadd_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Element-wise subtraction operations
 * ======================================== */

void _matsub(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matsub_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matsub_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Element-wise division operations
 * ======================================== */

void _matdiv(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matdiv_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matdiv_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Mathematical functions
 * ======================================== */

void _matexp(const float *m, float *res, unsigned int r, unsigned int c);

/* ========================================
 * Reduction operations
 * ======================================== */

void _matsum_rowwise(float *m, float *res, unsigned int r, unsigned int c);

void _matsum_colwise(float *m, float *res, unsigned int r, unsigned int c);

void _matmax_rowwise(float *m, float *res, unsigned int r, unsigned int c);

void _matmax_colwise(float *m, float *res, unsigned int r, unsigned int c);

/* ========================================
 * Matrix transformations
 * ======================================== */

void _mattranspose(const float *m, unsigned int r, unsigned int c, float *res);

/* ========================================
 * In-place scalar operations
 * ======================================== */

void _matscale(float *m, unsigned int r, unsigned int c, float alpha);

/* ========================================
 * Clamping operations
 * ======================================== */

void _matclampmin(float *m, unsigned int r, unsigned int c, float lo);

#endif // _RLM_MATRIX_OPS_H