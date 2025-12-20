/**
 * @file matrix_ops.c
 * @brief Implementation of low-level optimized matrix operations
 * 
 * This module provides performance-critical operations on raw float arrays.
 * Optimizations include:
 * - Loop unrolling (4x in matmul)
 * - Cache-friendly blocked transpose (8x8 blocks for 16KB L1 cache)
 * - Direct pointer arithmetic for minimal overhead
 */

#include <math.h>

#include "matrix_ops.h"

#define BLOCK 8  /**< Block size for cache-friendly transpose (Pentium II: 16KB L1) */

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#define min(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

/**
 * @brief Optimized matrix multiplication with 4x loop unrolling
 * 
 * Computes C = A * B where A is [r1 x c1] and B is [c1 x c2].
 * Uses loop unrolling to improve performance by reducing loop overhead
 * and enabling better instruction-level parallelism.
 */
void _matmul(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1, unsigned int c2) {
  unsigned int i, j, k;
  for (i = 0; i < r1; ++i) {
    const float * m1_row = m1 + i * c1;
    float * res_row = res + i * c2;
    for (j = 0; j < c2; ++j) {
      const float * a = m1_row;
      const float * b = m2 + j;
      float sum0 = 0.f;
      float sum1 = 0.f;
      float sum2 = 0.f;
      float sum3 = 0.f; /* main unrolled loop (4x) */
      for (k = 0; k + 3 < c1; k += 4) {
        sum0 += a[0] * b[0];
        sum1 += a[1] * b[c2];
        sum2 += a[2] * b[2 * c2];
        sum3 += a[3] * b[3 * c2];
        a += 4;
        b += 4 * c2;
      }
      float sum = sum0 + sum1 + sum2 + sum3; /* cleanup */
      for (; k < c1; ++k) {
        sum += ( * a) * ( * b);
        ++a;
        b += c2;
      }
      res_row[j] = sum;
    }
  }
}

/* ========================================
 * Element-wise addition operations
 * ======================================== */

void _matadd(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] + m2[i];
  }
}

void _matadd_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] + m2[i % c1];
  }
}

void _matadd_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] + m2[i / c1];
  }
}

/* ========================================
 * Element-wise subtraction operations
 * ======================================== */

void _matsub(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] - m2[i];
  }
}

void _matsub_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] - m2[i % c1];
  }
}

void _matsub_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] - m2[i / c1];
  }
}

/* ========================================
 * Element-wise division operations
 * ======================================== */

void _matdiv(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] / m2[i];
  }
}

void _matdiv_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] / m2[i % c1];
  }
}

void _matdiv_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1) {
  const unsigned int n = r1 * c1;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = m1[i] / m2[i / c1];
  }
}

/* ========================================
 * Mathematical functions
 * ======================================== */

/**
 * @brief Element-wise exponential using expf() for single precision
 */
void _matexp(const float *m, float *res, unsigned int r, unsigned int c) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    res[i] = expf(m[i]);
  }
}

/* ========================================
 * Reduction operations (sum/max along dimensions)
 * ======================================== */

void _matsum_rowwise(float *m, float *res, unsigned int r, unsigned int c) {
  for (unsigned int i = 0; i < r; ++i) {
    res[i] = 0.0;
    for (unsigned int j = 0; j < c; ++j) {
      res[i] += m[i*c + j];
    }
  }
}

void _matsum_colwise(float *m, float *res, unsigned int r, unsigned int c) {
  for (unsigned int i = 0; i < c; ++i) {
    res[i] = 0.0;
    for (unsigned int j = 0; j < r; ++j) {
      res[i] += m[j*c + i];
    }
  }
}

void _matmax_rowwise(float *m, float *res, unsigned int r, unsigned int c) {
  for (unsigned int i = 0; i < r; ++i) {
    res[i] = m[i*c];
    for (unsigned int j = 1; j < c; ++j) {
      if (m[i*c + j] > res[i]) {
        res[i] = m[i*c + j];
      }
    }
  }
}

void _matmax_colwise(float *m, float *res, unsigned int r, unsigned int c) {
  for (unsigned int i = 0; i < c; ++i) {
    res[i] = m[i];
    for (unsigned int j = 1; j < r; ++j) {
      if (m[j*c + i] > res[i]) {
        res[i] = m[j*c + i];
      }
    }
  }
}

/* ========================================
 * Matrix transpose with cache-friendly blocking
 * ======================================== */

/**
 * @brief Cache-optimized transpose using 8x8 blocks
 * 
 * Processes matrix in small blocks to maximize L1 cache hits.
 * Block size of 8 chosen for Pentium II architecture.
 */
void _mattranspose(const float *m, unsigned int r, unsigned int c, float *res) {
  unsigned int i, j, ii, jj;
  unsigned int i_max, j_max;
  
  for (ii = 0; ii < r; ii += BLOCK) {
      i_max = (ii + BLOCK < r) ? ii + BLOCK : r;
      for (jj = 0; jj < c; jj += BLOCK) {
          j_max = (jj + BLOCK < c) ? jj + BLOCK : c;
          for (i = ii; i < i_max; ++i)
              for (j = jj; j < j_max; ++j)
                  res[j*r + i] = m[i*c + j];
      }
  }
}

/* ========================================
 * In-place scalar operations
 * ======================================== */

void _matscale(float *m, unsigned int r, unsigned int c, float alpha) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    m[i] *= alpha;
  }
}

void _matshift(float *m, unsigned int r, unsigned int c, float beta) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    m[i] += beta;
  }
}

/* ========================================
 * Clamping operations
 * ======================================== */

void _matclamp(float *m, unsigned int r, unsigned int c, float lo, float hi) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = max(lo, m[i]);
    m[i] = min(hi, m[i]);
  }
}

void _matclampmin(float *m, unsigned int r, unsigned int c, float lo) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = max(lo, m[i]);
  }
}

void _matclampmax(float *m, unsigned int r, unsigned int c, float hi) {
  const unsigned int n = r * c;
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = min(hi, m[i]);
  }
}
