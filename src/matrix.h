#ifndef _RLM_MATRIX_H
#define _RLM_MATRIX_H

struct Matrix2D {
	unsigned int r;
	unsigned int c;
	float* data;
};

struct Matrix2D new_mat(unsigned int r, unsigned int c);

void del_mat(struct Matrix2D *m);

float* mat_at(const struct Matrix2D *m, unsigned int i, unsigned int j);

void print_mat(const struct Matrix2D *m);

void matmul(const struct Matrix2D *m1, const struct Matrix2D *m2, struct Matrix2D *res);


#endif // _RLM_MATRIX_H