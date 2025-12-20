/**
 * @file matrix.h
 * @brief High-level 2D matrix API for float and unsigned int matrices
 * 
 * This module provides a comprehensive matrix API for neural network computations.
 * It defines matrix structures and operations including:
 * - Construction/destruction
 * - Element access
 * - Linear algebra (multiplication, addition, subtraction, etc.)
 * - Scalar operations
 * - Matrix transformations
 * - Utility functions
 * 
 * All operations return new matrices unless explicitly marked as in-place.
 * Matrices use row-major storage order.
 */

#ifndef _RLM_MATRIX_H
#define _RLM_MATRIX_H

/**
 * @brief 2D matrix structure for float data
 * 
 * Stores a matrix in row-major order. Element (i,j) is at data[i*c + j].
 */
struct Matrix2D {
	unsigned int r;    /**< Number of rows */
	unsigned int c;    /**< Number of columns */
	float* data;       /**< Flat array in row-major order */
};

/**
 * @brief 2D matrix structure for unsigned int data
 * 
 * Used primarily for storing token indices and discrete values.
 */
struct Matrix2D_UInt {
	unsigned int r;    /**< Number of rows */
	unsigned int c;    /**< Number of columns */
	unsigned int* data; /**< Flat array in row-major order */
};

/* ========================================
 * Construction and Destruction
 * ======================================== */

/**
 * @brief Create a new float matrix with zero-initialized data
 * 
 * @param r Number of rows (must be > 0)
 * @param c Number of columns (must be > 0)
 * @return Matrix2D structure with allocated and zeroed data
 * 
 * @throws InvalidInput if r == 0 or c == 0
 * @throws MemoryError if allocation fails
 * @note Must be freed with mat_free()
 */
struct Matrix2D mat_new(unsigned int r, unsigned int c);

/**
 * @brief Free matrix memory and reset fields
 * 
 * @param m Pointer to matrix to free
 * 
 * @note Sets data to NULL and dimensions to 0 after freeing
 * @note Safe to call multiple times
 */
void mat_free(struct Matrix2D *m);

/**
 * @brief Create a new unsigned int matrix with zero-initialized data
 * 
 * @param r Number of rows (must be > 0)
 * @param c Number of columns (must be > 0)
 * @return Matrix2D_UInt structure with allocated and zeroed data
 * 
 * @throws InvalidInput if r == 0 or c == 0
 * @throws MemoryError if allocation fails
 * @note Must be freed with mat_uint_free()
 */
struct Matrix2D_UInt mat_uint_new(unsigned int r, unsigned int c);

/**
 * @brief Free unsigned int matrix memory and reset fields
 * 
 * @param m Pointer to matrix to free
 */
void mat_uint_free(struct Matrix2D_UInt *m);

/**
 * @brief Create a row vector of sequential indices [0, 1, 2, ..., n-1]
 * 
 * @param n Number of indices
 * @return Matrix2D_UInt of shape [1 x n] with values 0 to n-1
 * 
 * @throws InvalidInput if n == 0
 * @note Useful for creating positional embeddings indices
 */
struct Matrix2D_UInt indices_new(unsigned int n);

/* ========================================
 * Element Access and Manipulation
 * ======================================== */

/**
 * @brief Get pointer to element at position (i, j)
 * 
 * @param m Pointer to matrix
 * @param i Row index (0-based)
 * @param j Column index (0-based)
 * @return Pointer to element at (i, j)
 * 
 * @throws IndexError if index is out of bounds
 * @note Allows both reading and writing: *mat_at(&m, i, j) = value;
 */
float *mat_at(const struct Matrix2D *m,
              unsigned int i,
              unsigned int j);

/**
 * @brief Mask upper triangle of matrix with specified value
 * 
 * Sets all elements above the diagonal to the given value.
 * Used for causal attention masking where m[i,j] = value for j > i.
 * 
 * @param m Pointer to matrix to modify (in-place)
 * @param value Value to set in upper triangle (typically -inf)
 * 
 * @note Only modifies strict upper triangle (not the diagonal)
 */
void mat_maskdiag(struct Matrix2D *m, float value);

/* ========================================
 * Utilities
 * ======================================== */

/**
 * @brief Print matrix contents to stdout via logger
 * 
 * For large matrices (> 100 elements), prints only a 5x10 corner.
 * For small matrices, prints all values.
 * 
 * @param m Pointer to matrix to print
 * 
 * @note Uses logger at INFO level
 * @note Format: 4 decimal places per element
 */
void mat_print(const struct Matrix2D *m);

/* ========================================
 * Core Linear Algebra Operations
 * ======================================== */

/**
 * @brief Matrix multiplication: C = A * B
 * 
 * @param m1 Left matrix [r1 x c1]
 * @param m2 Right matrix [c1 x c2]
 * @return New matrix [r1 x c2]
 * 
 * @throws InvalidInput if dimensions don't match (m1.c != m2.r)
 * @note Returned matrix must be freed by caller
 */
struct Matrix2D mat_mul(const struct Matrix2D *m1, const struct Matrix2D *m2);

/**
 * @brief Element-wise division with broadcasting support
 * 
 * Supports:
 * - Same dimensions: element-wise division
 * - Row broadcast: m2 is [1 x c], broadcasted across rows
 * - Column broadcast: m2 is [r x 1], broadcasted across columns
 * 
 * @param m1 Dividend matrix
 * @param m2 Divisor matrix
 * @return New matrix with division result
 * 
 * @throws InvalidInput if dimensions incompatible for broadcasting
 */
struct Matrix2D mat_div(const struct Matrix2D *m1, const struct Matrix2D *m2);

/**
 * @brief Element-wise addition with broadcasting support
 * 
 * @see mat_div for broadcasting rules
 */
struct Matrix2D mat_add(const struct Matrix2D *m1, const struct Matrix2D *m2);

/**
 * @brief Element-wise subtraction with broadcasting support
 * 
 * @see mat_div for broadcasting rules
 */
struct Matrix2D mat_sub(const struct Matrix2D *m1, const struct Matrix2D *m2);

/**
 * @brief Element-wise exponential: result[i] = exp(m[i])
 * 
 * @param m Input matrix
 * @return New matrix with exponentials
 */
struct Matrix2D mat_exp(const struct Matrix2D *m);

/**
 * @brief Sum along specified dimension
 * 
 * @param m Input matrix [r x c]
 * @param dim Dimension to sum: 0 for column-wise, 1 for row-wise
 * @return New matrix: [1 x c] if dim=0, [r x 1] if dim=1
 * 
 * @throws InvalidInput if dim > 1
 */
struct Matrix2D mat_sum(const struct Matrix2D *m, unsigned short dim);

/**
 * @brief Maximum along specified dimension
 * 
 * @param m Input matrix [r x c]
 * @param dim Dimension for max: 0 for column-wise, 1 for row-wise
 * @return New matrix: [1 x c] if dim=0, [r x 1] if dim=1
 * 
 * @throws InvalidInput if dim > 1
 */
struct Matrix2D mat_max(const struct Matrix2D *m, unsigned short dim);

/* ========================================
 * In-Place Scalar Operations
 * ======================================== */

/**
 * @brief Scale matrix in-place: m = m * alpha
 * 
 * @param m Pointer to matrix to modify
 * @param alpha Scaling factor
 */
void mat_scale(struct Matrix2D *m, float alpha);

/**
 * @brief Shift matrix in-place: m = m + beta
 * 
 * @param m Pointer to matrix to modify
 * @param beta Value to add to all elements
 */
void mat_shift(struct Matrix2D *m, float beta);

/* ========================================
 * Matrix Transformations
 * ======================================== */

/**
 * @brief Transpose matrix: result[j,i] = m[i,j]
 * 
 * @param m Input matrix [r x c]
 * @return New transposed matrix [c x r]
 * 
 * @note Uses cache-friendly blocked algorithm
 */
struct Matrix2D mat_transpose(const struct Matrix2D *m);

/**
 * @brief Create identity matrix
 * 
 * @param n Dimension of square identity matrix
 * @return New n x n matrix with 1s on diagonal, 0s elsewhere
 */
struct Matrix2D mat_identity(unsigned int n);

/* ========================================
 * Copy and Clamping Operations
 * ======================================== */

/**
 * @brief Create a deep copy of matrix
 * 
 * @param m Matrix to copy
 * @return New matrix with copied data
 */
struct Matrix2D mat_copy(const struct Matrix2D *m);

/**
 * @brief Clamp matrix values to range [lo, hi]
 * 
 * @param m Input matrix
 * @param lo Lower bound
 * @param hi Upper bound (must be > lo)
 * @return New matrix with clamped values
 * 
 * @throws InvalidInput if lo >= hi
 */
struct Matrix2D mat_clamp(const struct Matrix2D *m, float lo, float hi);

/**
 * @brief Clamp matrix values to minimum: result = max(m, lo)
 * 
 * @param m Input matrix
 * @param lo Minimum value
 * @return New matrix with clamped values
 */
struct Matrix2D mat_clamp_min(const struct Matrix2D *m, float lo);

/**
 * @brief Clamp matrix values to maximum: result = min(m, hi)
 * 
 * @param m Input matrix
 * @param hi Maximum value
 * @return New matrix with clamped values
 */
struct Matrix2D mat_clamp_max(const struct Matrix2D *m, float hi);

/**
 * @brief Select rows from matrix by indices
 * 
 * Performs advanced indexing: result[i] = m[indices[i]]
 * Used for embedding lookups.
 * 
 * @param m Input matrix [r x c]
 * @param indices Row indices to select [1 x n]
 * @return New matrix [n x c] with selected rows
 * 
 * @throws InvalidInput if indices.r != 1
 * @throws InvalidInput if any index >= m.r
 * @throws InvalidInput if n_indices > m.r
 */
struct Matrix2D mat_rowselect(const struct Matrix2D *m, const struct Matrix2D_UInt *indices);

/**
 * @brief Initialize matrix with random values in [0, 1]
 * 
 * @param m Pointer to matrix to initialize (in-place)
 * 
 * @note Uses rand() function, so seed with srand() first
 */
void mat_random_init(struct Matrix2D *m);

#endif // _RLM_MATRIX_H