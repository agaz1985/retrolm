#include <stdlib.h>
#include <string.h>

#include "memory.h"

float* alloc_mat(unsigned int r, unsigned int c) {
	float *m = (float*)malloc(r * c * sizeof(float));
	memset(m, 0, r * c * sizeof(float));
	return m;
}

void free_mat(float *m) {
	free(m);
}


