#include <stdlib.h>
#include <string.h>
#include "memory.h"
#include "exceptions.h"

float* alloc_mat_float(unsigned int r, unsigned int c) {
	float *m = (float*)malloc(r * c * sizeof(float));
	if (m == NULL) {
		throw("Failed to allocate memory for float matrix.\n", MemoryError);
	}
	memset(m, 0, r * c * sizeof(float));
	return m;
}

unsigned int* alloc_mat_uint(unsigned int r, unsigned int c) {
	unsigned int *m = (unsigned int*)malloc(r * c * sizeof(unsigned int));
	if (m == NULL) {
		throw("Failed to allocate memory for uint matrix.\n", MemoryError);
	}
	memset(m, 0, r * c * sizeof(unsigned int));
	return m;
}

void free_mat_float(float *m) {
	if (m != NULL) {  // Good practice to check before free
		free(m);
	}
}

void free_mat_uint(unsigned int *m) {
	if (m != NULL) {  // Good practice to check before free
		free(m);
	}
}