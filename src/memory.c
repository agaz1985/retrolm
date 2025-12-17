#include <stdlib.h>
#include <string.h>

#include "memory.h"

float* alloc_mat_float(unsigned int r, unsigned int c) {
	float *m = (float*)malloc(r * c * sizeof(float));
	memset(m, 0, r * c * sizeof(float));
	return m;
}

unsigned int* alloc_mat_uint(unsigned int r, unsigned int c) {
	unsigned int *m = (unsigned int*)malloc(r * c * sizeof(unsigned int));
	memset(m, 0, r * c * sizeof(unsigned int));
	return m;
}

void free_mat_float(float *m) {
	free(m);
}

void free_mat_uint(unsigned int *m) {
	free(m);
}


