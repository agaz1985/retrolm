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

struct Matrix2D mat_div(const struct Matrix2D *m1, const struct Matrix2D *m2);
struct Matrix2D mat_add(const struct Matrix2D *m1, const struct Matrix2D *m2);
struct Matrix2D mat_exp(const struct Matrix2D *m);
struct Matrix2D mat_sum(const struct Matrix2D *m, unsigned short dim);

/* scalar operations (in-place) */
void mat_scale(struct Matrix2D *m, float alpha);
void mat_shift(struct Matrix2D *m, float beta);

/* special matrices and transforms */
struct Matrix2D mat_transpose(const struct Matrix2D *m);

struct Matrix2D mat_identity(unsigned int n);

/* copy and clamping operations */

struct Matrix2D mat_copy(const struct Matrix2D *m);

struct Matrix2D mat_clamp(const struct Matrix2D *m, float lo, float hi);
struct Matrix2D mat_clamp_min(const struct Matrix2D *m, float lo);
struct Matrix2D mat_clamp_max(const struct Matrix2D *m, float hi);

void mat_random_init(struct Matrix2D *m);

#endif // _RLM_MATRIX_H