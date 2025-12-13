#ifndef _RLM_MATRIX_H
#define _RLM_MATRIX_H

struct Matrix2D {
	unsigned int r;
	unsigned int c;
	float** data;
};

struct Matrix2D new_mat(unsigned int r, unsigned int c);
void del_mat(struct Matrix2D m);

#endif // _RLM_MATRIX_H