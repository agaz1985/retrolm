#include <stdlib.h>
#include <string.h>

#include "memory.h"

float** alloc_mat(unsigned int r, unsigned int c) {
	float **m;
	m = malloc(r * sizeof(*m));
	for (int i = 0; i < r; ++i) {
		m[i] = malloc(c * sizeof(*m[i]));
		memset(m[i], 0, c * sizeof(*m[i]));
	}
	return m;
}

void free_mat(float** m, unsigned int r, unsigned int c) {
	for (int i = 0; i < r; ++i) {
		free(m[i]);
	}
	free(m);
}