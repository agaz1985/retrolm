/**
 * @file matrix_ops.h
 * @brief Low-level matrix operations on raw float arrays
 * 
 * This module provides optimized low-level operations on flat float arrays representing
 * matrices in row-major order. These functions work directly with pointers for performance
 * and are used by the higher-level matrix.h API.
 * 
 * All operations assume row-major layout: data[i*cols + j] for element (i,j).
 * Broadcasting operations handle dimension mismatches by repeating along rows or columns.
 */

#ifndef _RLM_MATRIX_OPS_H
#define _RLM_MATRIX_OPS_H

/**
 * @brief Matrix multiplication: res = m1 * m2
 * 
 * Optimized implementation with loop unrolling (4x) for better performance.
 * 
 * @param m1 Left matrix [r1 x c1]
 * @param m2 Right matrix [c1 x c2]
 * @param res Output matrix [r1 x c2]
 * @param r1 Number of rows in m1
 * @param c1 Number of columns in m1 (must equal rows in m2)
 * @param c2 Number of columns in m2
 */
void _matmul(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1, unsigned int c2);

/* ========================================
 * Element-wise addition operations
 * ======================================== */

/**
 * @brief Element-wise addition: res = m1 + m2
 * 
 * @param m1 First matrix [r1 x c1]
 * @param m2 Second matrix [r1 x c1]
 * @param res Output matrix [r1 x c1]
 * @param r1 Number of rows
 * @param c1 Number of columns
 */
void _matadd(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Addition with row broadcasting: res[i,j] = m1[i,j] + m2[j]
 * 
 * Broadcasts m2 (row vector [1 x c1]) across all rows of m1.
 * 
 * @param m1 Matrix [r1 x c1]
 * @param m2 Row vector [1 x c1]
 * @param res Output matrix [r1 x c1]
 * @param r1 Number of rows
 * @param c1 Number of columns
 */
void _matadd_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Addition with column broadcasting: res[i,j] = m1[i,j] + m2[i]
 * 
 * Broadcasts m2 (column vector [r1 x 1]) across all columns of m1.
 * 
 * @param m1 Matrix [r1 x c1]
 * @param m2 Column vector [r1 x 1]
 * @param res Output matrix [r1 x c1]
 * @param r1 Number of rows
 * @param c1 Number of columns
 */
void _matadd_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Element-wise subtraction operations
 * ======================================== */

/**
 * @brief Element-wise subtraction: res = m1 - m2
 */
void _matsub(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Subtraction with row broadcasting: res[i,j] = m1[i,j] - m2[j]
 */
void _matsub_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Subtraction with column broadcasting: res[i,j] = m1[i,j] - m2[i]
 */
void _matsub_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Element-wise division operations
 * ======================================== */

/**
 * @brief Element-wise division: res = m1 / m2
 */
void _matdiv(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Division with row broadcasting: res[i,j] = m1[i,j] / m2[j]
 */
void _matdiv_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/**
 * @brief Division with column broadcasting: res[i,j] = m1[i,j] / m2[i]
 */
void _matdiv_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

/* ========================================
 * Mathematical functions
 * ======================================== */

/**
 * @brief Element-wise exponential: res[i] = exp(m[i])
 * 
 * @param m Input matrix
 * @param res Output matrix (same dimensions)
 * @param r Number of rows
 * @param c Number of columns
 */
void _matexp(const float *m, float *res, unsigned int r, unsigned int c);

/* ========================================
 * Reduction operations
 * ======================================== */

/**
 * @brief Sum along rows: res[i] = sum(m[i,:])
 * 
 * @param m Input matrix [r x c]
 * @param res Output vector [r x 1]
 * @param r Number of rows
 * @param c Number of columns
 */
void _matsum_rowwise(float *m, float *res, unsigned int r, unsigned int c);

/**
 * @brief Sum along columns: res[j] = sum(m[:,j])
 * 
 * @param m Input matrix [r x c]
 * @param res Output vector [1 x c]
 * @param r Number of rows
 * @param c Number of columns
 */
void _matsum_colwise(float *m, float *res, unsigned int r, unsigned int c);

/**
 * @brief Max along rows: res[i] = max(m[i,:])
 * 
 * @param m Input matrix [r x c]
 * @param res Output vector [r x 1]
 * @param r Number of rows
 * @param c Number of columns
 */
void _matmax_rowwise(float *m, float *res, unsigned int r, unsigned int c);

/**
 * @brief Max along columns: res[j] = max(m[:,j])
 * 
 * @param m Input matrix [r x c]
 * @param res Output vector [1 x c]
 * @param r Number of rows
 * @param c Number of columns
 */
void _matmax_colwise(float *m, float *res, unsigned int r, unsigned int c);

/* ========================================
 * Matrix transformations
 * ======================================== */

/**
 * @brief Transpose matrix: res[j,i] = m[i,j]
 * 
 * Uses cache-friendly blocked algorithm (8x8 blocks) optimized for
 * Pentium II's 16KB L1 cache.
 * 
 * @param m Input matrix [r x c]
 * @param r Number of rows in input
 * @param c Number of columns in input
 * @param res Output matrix [c x r]
 */
void _mattranspose(const float *m, unsigned int r, unsigned int c, float *res);

/* ========================================
 * In-place scalar operations
 * ======================================== */

/**
 * @brief Scale matrix in-place: m[i] *= alpha
 * 
 * @param m Matrix to modify [r x c]
 * @param r Number of rows
 * @param c Number of columns
 * @param alpha Scaling factor
 */
void _matscale(float *m, unsigned int r, unsigned int c, float alpha);

/**
 * @brief Shift matrix in-place: m[i] += beta
 * 
 * @param m Matrix to modify [r x c]
 * @param r Number of rows
 * @param c Number of columns
 * @param beta Value to add
 */
void _matshift(float *m, unsigned int r, unsigned int c, float beta);

/* ========================================
 * Clamping operations
 * ======================================== */

/**
 * @brief Clamp values in-place: m[i] = clamp(m[i], lo, hi)
 * 
 * @param m Matrix to modify
 * @param r Number of rows
 * @param c Number of columns
 * @param lo Lower bound
 * @param hi Upper bound
 */
void _matclamp(float *m, unsigned int r, unsigned int c, float lo, float hi);

/**
 * @brief Clamp minimum in-place: m[i] = max(m[i], lo)
 */
void _matclampmin(float *m, unsigned int r, unsigned int c, float lo);

/**
 * @brief Clamp maximum in-place: m[i] = min(m[i], hi)
 */
void _matclampmax(float *m, unsigned int r, unsigned int c, float hi);

#endif // _RLM_MATRIX_OPS_H