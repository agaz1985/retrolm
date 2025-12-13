#include <stdio.h>

#include "matrix.h"
#include "memory.h"

struct Matrix2D new_mat(unsigned int r, unsigned int c) {
	float **data = alloc_mat(r, c);
	struct Matrix2D m = {r, c, data};
	return m;
}

void del_mat(struct Matrix2D m) {
	free_mat(m.data, m.r);
	m.data = NULL;
	m.r = 0;
	m.c = 0;
}
