#include <stdio.h>

#include "logger.h"
#include "matrix.h"
#include "memory.h"
#include "exceptions.h"
#include "matrix_ops.h"

struct Matrix2D mat_new(unsigned int r, unsigned int c) {
	if (r == 0) {
		throw("Matrix number of rows cannot be zero.", InvalidInput);
	}
	if (c == 0) {
		throw("Matrix number of columns cannot be zero.", InvalidInput);
	}

	float *data = alloc_mat(r, c);
	struct Matrix2D m = {r, c, data};
	return m;
}

void mat_free(struct Matrix2D *m) {
	free_mat(m->data);
	m->data = NULL;
	m->r = 0;
	m->c = 0;
}

float* mat_at(const struct Matrix2D *m, unsigned int i, unsigned int j) {
	const unsigned int index = i * m->c + j;

	if (index >= m->r * m->c) {
		throw("Matrix index out of range.", IndexError);
	}

	return &m->data[index];
}

void mat_print(const struct Matrix2D *m)
{
    unsigned int i, j;
    char buffer[4096];
    int offset = 0;

    offset += sprintf(buffer + offset, "\n");

    for (i = 0; i < m->r; ++i) {
        for (j = 0; j < m->c; ++j) {
            offset += sprintf(
                buffer + offset,
                "%f,",
                *mat_at(m, i, j)
            );
        }
        offset += sprintf(buffer + offset, "\n");
    }

    logger(buffer, INFO);
}

// Matrix Operations

struct Matrix2D mat_mul(const struct Matrix2D *m1, const struct Matrix2D *m2) {
  if (m1->c != m2->r) {
    throw ("Matrix dimensions do not match!\n", InvalidInput);
  }

  struct Matrix2D res = mat_new(m1->r, m2->c);
  _matmul(m1->data, m2->data, res.data, m1->r, m1->c, m2->c);
  return res;
}

/* scalar operations (in-place) */
void mat_scale(struct Matrix2D *m, float alpha) {
	_matscale(m->data, m->r * m->c, alpha);
}

void mat_shift(struct Matrix2D *m, float beta) {
	_matshift(m->data, m->r * m->c, beta);
}

/* special matrices / transforms */
struct Matrix2D mat_transpose(const struct Matrix2D *m) {
	struct Matrix2D res = mat_new(m->c, m->r);
	_mattranspose(m->data, m->r, m->c, res.data);
	return res;
}

struct Matrix2D mat_identity(unsigned int n) {
	struct Matrix2D res = mat_new(n, n);
	for (unsigned int i = 0; i < n; ++i) {
		*mat_at(&res, i, i) = 1.f;
	}
	return res;
}
