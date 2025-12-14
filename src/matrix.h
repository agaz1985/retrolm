#ifndef _RLM_MATRIX_H
#define _RLM_MATRIX_H

struct Matrix2D {
	unsigned int r;
	unsigned int c;
	float* data;
};

/* construction / destruction */
struct Matrix2D mat_new(unsigned int r, unsigned int c);
void mat_free(struct Matrix2D *m);

/* element access */
float *mat_at(const struct Matrix2D *m,
              unsigned int i,
              unsigned int j);

/* utilities */
void mat_print(const struct Matrix2D *m);

/* core linear algebra */
struct Matrix2D mat_mul(const struct Matrix2D *m1, const struct Matrix2D *m2);

/* scalar operations (in-place) */
void mat_scale(struct Matrix2D *m, float alpha);
void mat_shift(struct Matrix2D *m, float beta);

/* special matrices / transforms */
struct Matrix2D mat_transpose(const struct Matrix2D *m);

struct Matrix2D mat_identity(unsigned int n);

#endif // _RLM_MATRIX_H