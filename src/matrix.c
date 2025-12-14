#include <stdio.h>

#include "logger.h"
#include "matrix.h"
#include "memory.h"
#include "exceptions.h"
#include "matrix_ops.h"

struct Matrix2D new_mat(unsigned int r, unsigned int c) {
	float *data = alloc_mat(r, c);
	struct Matrix2D m = {r, c, data};
	return m;
}

void del_mat(struct Matrix2D *m) {
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

void print_mat(const struct Matrix2D *m)
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

void matmul(const struct Matrix2D * m1,
  const struct Matrix2D * m2, struct Matrix2D * res) {
  if (m1 -> c != m2 -> r) {
    throw ("Matrix dimensions do not match!\n", InvalidInput);
  }

  _matmul(m1 -> data, m2 -> data, res -> data, m1 -> r, m1 -> c, m2 -> c);
}
