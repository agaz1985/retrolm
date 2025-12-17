#ifndef _RLM_MATRIX_OPS_H
#define _RLM_MATRIX_OPS_H

void _matmul(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1, unsigned int c2);

void _matadd(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matadd_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matadd_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matsub(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matsub_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matsub_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matdiv(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matdiv_rowbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);
void _matdiv_colbroadcast(const float *m1, const float *m2, float *res, unsigned int r1, unsigned int c1);

void _matexp(const float *m, float *res, unsigned int r, unsigned int c);

void _matsum_rowwise(float *m, float *res, unsigned int r, unsigned int c);
void _matsum_colwise(float *m, float *res, unsigned int r, unsigned int c);

void _matmax_rowwise(float *m, float *res, unsigned int r, unsigned int c);
void _matmax_colwise(float *m, float *res, unsigned int r, unsigned int c);

void _mattranspose(const float *m, unsigned int r, unsigned int c, float *res);

void _matscale(float *m, unsigned int r, unsigned int c, float alpha);
void _matshift(float *m, unsigned int r, unsigned int c, float beta);

void _matclamp(float *m, unsigned int r, unsigned int c, float lo, float hi);
void _matclampmin(float *m, unsigned int r, unsigned int c, float lo);
void _matclampmax(float *m, unsigned int r, unsigned int c, float hi);

#endif // _RLM_MATRIX_OPS_H