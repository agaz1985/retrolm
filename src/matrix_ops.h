#ifndef _RLM_MATRIX_OPS_H
#define _RLM_MATRIX_OPS_H

void _matmul(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1, unsigned int c2);

void _mattranspose(const float *m, unsigned int r, unsigned int c, float *res);

void _matscale(float *m, unsigned int n, float alpha);

void _matshift(float *m, unsigned int n, float beta);

#endif // _RLM_MATRIX_OPS_H