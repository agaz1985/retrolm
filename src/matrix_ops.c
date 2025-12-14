#include "matrix_ops.h"

#define BLOCK 8 // Small block for 16KB L1 cache of Pentium II

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#define min(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

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

void _matscale(float *m, unsigned int n, float alpha) {
  for (unsigned int i = 0; i < n; ++i) {
    m[i] *= alpha;
  }
}

void _matshift(float *m, unsigned int n, float beta) {
  for (unsigned int i = 0; i < n; ++i) {
    m[i] += beta;
  }
}

void _matclamp(float *m, unsigned int n, float lo, float hi) {
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = max(lo, m[i]);
    m[i] = min(hi, m[i]);
  }
}

void _matclampmin(float *m, unsigned int n, float lo) {
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = max(lo, m[i]);
  }
}

void _matclampmax(float *m, unsigned int n, float hi) {
  for (unsigned int i = 0; i < n; ++i) {
    m[i] = min(hi, m[i]);
  }
}
