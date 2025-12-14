#include "matrix_ops.h"

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
