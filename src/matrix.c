#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "logger.h"
#include "matrix.h"
#include "memory.h"
#include "exceptions.h"
#include "matrix_ops.h"

struct Matrix2D mat_new(unsigned int r, unsigned int c) {
	if (r == 0) {
		throw("Matrix number of rows cannot be zero.\n", InvalidInput);
	}
	if (c == 0) {
		throw("Matrix number of columns cannot be zero.\n", InvalidInput);
	}

	float *data = alloc_mat_float(r, c);
	struct Matrix2D m = {r, c, data};
	return m;
}

void mat_free(struct Matrix2D *m) {
	free_mat_float(m->data);
	m->data = NULL;
	m->r = 0;
	m->c = 0;
}

struct Matrix2D_UInt mat_uint_new(unsigned int r, unsigned int c) {
	if (r == 0) {
		throw("Matrix number of rows cannot be zero.\n", InvalidInput);
	}
	if (c == 0) {
		throw("Matrix number of columns cannot be zero.\n", InvalidInput);
	}

	unsigned int *data = alloc_mat_uint(r, c);
	struct Matrix2D_UInt m = {r, c, data};
	return m;
}

void mat_uint_free(struct Matrix2D_UInt *m) {
	free_mat_uint(m->data);
	m->data = NULL;
	m->r = 0;
	m->c = 0;
}

struct Matrix2D_UInt indices_new(unsigned int n) {
	if (n == 0) {
		throw("Number of indices cannot be zero.\n", InvalidInput);
	}
	struct Matrix2D_UInt indices = mat_uint_new(1, n);
	for (unsigned int i = 0; i < n; ++i) {
		indices.data[i] = i;
	}
	return indices;
}

float* mat_at(const struct Matrix2D *m, unsigned int i, unsigned int j) {
	const unsigned int index = i * m->c + j;

	if (index >= m->r * m->c) {
		throw("Matrix index out of range.\n", IndexError);
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

struct Matrix2D mat_div(const struct Matrix2D *m1, const struct Matrix2D *m2) {
  if (m2->r == 1 && m1->c != m2->c) {
    throw ("Unable to broadcast, matrix columns do not match!\n", InvalidInput);  	
  }
  if (m2->c == 1 && m1->r != m2->r) {
    throw("Unable to broadcast, matrix rows do not match!\n", InvalidInput);  	
  }
  if (m2->r != 1 && m1->r != m2->r) {
    throw("Matrix rows do not match!\n", InvalidInput);
  }
  if (m2->c != 1 && m1->c != m2->c) {
    throw("Matrix columns do not match!\n", InvalidInput);
  }

  struct Matrix2D res = mat_new(m1->r, m1->c);

  if ((m1->r == m2->r) && (m1->c == m2->c)) {
  	_matdiv(m1->data, m2->data, res.data, m1->r, m1->c);
  } else if (m2->r == 1) {
  	_matdiv_rowbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  } else {
  	assert(m2->c == 1);
  	_matdiv_colbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  }
  return res;
}

struct Matrix2D mat_add(const struct Matrix2D *m1, const struct Matrix2D *m2) {
  if (m2->r == 1 && m1->c != m2->c) {
    throw ("Unable to broadcast, matrix columns do not match!\n", InvalidInput);  	
  }
  if (m2->c == 1 && m1->r != m2->r) {
    throw("Unable to broadcast, matrix rows do not match!\n", InvalidInput);  	
  }
  if (m2->r != 1 && m1->r != m2->r) {
    throw("Matrix rows do not match!\n", InvalidInput);
  }
  if (m2->c != 1 && m1->c != m2->c) {
    throw("Matrix columns do not match!\n", InvalidInput);
  }

  struct Matrix2D res = mat_new(m1->r, m1->c);

  if ((m1->r == m2->r) && (m1->c == m2->c)) {
  	_matadd(m1->data, m2->data, res.data, m1->r, m1->c);
  } else if (m2->r == 1) {
  	_matadd_rowbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  } else {
  	assert(m2->c == 1);
  	_matadd_colbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  }
  return res;
}

struct Matrix2D mat_sub(const struct Matrix2D *m1, const struct Matrix2D *m2) {
  if (m2->r == 1 && m1->c != m2->c) {
    throw ("Unable to broadcast, matrix columns do not match!\n", InvalidInput);  	
  }
  if (m2->c == 1 && m1->r != m2->r) {
    throw("Unable to broadcast, matrix rows do not match!\n", InvalidInput);  	
  }
  if (m2->r != 1 && m1->r != m2->r) {
    throw("Matrix rows do not match!\n", InvalidInput);
  }
  if (m2->c != 1 && m1->c != m2->c) {
    throw("Matrix columns do not match!\n", InvalidInput);
  }

  struct Matrix2D res = mat_new(m1->r, m1->c);

  if ((m1->r == m2->r) && (m1->c == m2->c)) {
  	_matsub(m1->data, m2->data, res.data, m1->r, m1->c);
  } else if (m2->r == 1) {
  	_matsub_rowbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  } else {
  	assert(m2->c == 1);
  	_matsub_colbroadcast(m1->data, m2->data, res.data, m1->r, m1->c);
  }
  return res;
}

struct Matrix2D mat_exp(const struct Matrix2D *m) {
	struct Matrix2D res = mat_new(m->r, m->c);
	_matexp(m->data, res.data, m->r, m->c);
	return res;
}

struct Matrix2D mat_sum(const struct Matrix2D *m, unsigned short dim) {
	if (dim > 1) {
		throw("Invalid matrix dimension!\n", InvalidInput);
	}

	struct Matrix2D res;
	if (dim == 0) {
		res = mat_new(1, m->c);
		_matsum_colwise(m->data, res.data, m->r, m->c);
	} else {
		res = mat_new(m->r, 1);
		_matsum_rowwise(m->data, res.data, m->r, m->c);
	}
	return res;
}

struct Matrix2D mat_max(const struct Matrix2D *m, unsigned short dim) {
	if (dim > 1) {
		throw("Invalid matrix dimension!\n", InvalidInput);
	}

	struct Matrix2D res;
	if (dim == 0) {
		res = mat_new(1, m->c);
		_matmax_colwise(m->data, res.data, m->r, m->c);
	} else {
		res = mat_new(m->r, 1);
		_matmax_rowwise(m->data, res.data, m->r, m->c);
	}
	return res;	
}

/* scalar operations (in-place) */
void mat_scale(struct Matrix2D *m, float alpha) {
	_matscale(m->data, m->r, m->c, alpha);
}

void mat_shift(struct Matrix2D *m, float beta) {
	_matshift(m->data, m->r, m->c, beta);
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

/* copy and clamping operations */

struct Matrix2D mat_copy(const struct Matrix2D *m) {
	struct Matrix2D res = mat_new(m->r, m->c);
	memcpy(res.data, m->data, m->r * m->c * sizeof(float));
	return res;
}

struct Matrix2D mat_clamp(const struct Matrix2D *m, float lo, float hi) {
	if (lo >= hi) {
		throw("Low value must be strictly lower than the high value!\n", InvalidInput);
	}

	struct Matrix2D res = mat_copy(m);
	_matclamp(res.data, res.r, res.c, lo, hi);
	return res;
}

struct Matrix2D mat_clamp_min(const struct Matrix2D *m, float lo) {
	struct Matrix2D res = mat_copy(m);
	_matclampmin(res.data, res.r, res.c, lo);
	return res;
}

struct Matrix2D mat_clamp_max(const struct Matrix2D *m, float hi) {
	struct Matrix2D res = mat_copy(m);
	_matclampmax(res.data, res.r, res.c, hi);
	return res;
}

struct Matrix2D mat_rowselect(const struct Matrix2D *m, const struct Matrix2D_UInt *indices) {
	if (indices->r != 1) {
		throw("Number of rows must be equal to 1.\n", InvalidInput);
	}

	const unsigned int n_indices = indices->c;
	if (n_indices > m->r) {
		throw("The number of requested indices is higher than the number of matrix rows!\n", InvalidInput);
	}
  for (unsigned int i = 0; i < n_indices; ++i) {
      if (indices->data[i] >= m->r) {
          throw("Index out of bounds!\n", InvalidInput);
      }
  }

	struct Matrix2D res = mat_new(n_indices, m->c);
	for (unsigned int i = 0; i < n_indices; ++i) {
		memcpy(res.data + i*m->c, m->data + indices->data[i]*m->c, m->c * sizeof(float));
	}
	return res;
}

void mat_random_init(struct Matrix2D *m) {
	const unsigned int n = m->r * m->c;
	for (unsigned int i = 0; i < n; ++i) {
		m->data[i] = (float)rand() / RAND_MAX;
	}
}
