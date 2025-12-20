/**
 * @file memory.c
 * @brief Implementation of memory allocation wrappers
 */

#include <stdlib.h>
#include <string.h>
#include "memory.h"
#include "exceptions.h"

/**
 * @brief Allocate float array with error checking and zero-initialization
 */
float* alloc_mat_float(unsigned int r, unsigned int c) {
	float *m = (float*)malloc(r * c * sizeof(float));
	if (m == NULL) {
		throw("Failed to allocate memory for float matrix.\n", MemoryError);
	}
	memset(m, 0, r * c * sizeof(float));
	return m;
}

/**
 * @brief Allocate uint array with error checking and zero-initialization
 */
unsigned int* alloc_mat_uint(unsigned int r, unsigned int c) {
	unsigned int *m = (unsigned int*)malloc(r * c * sizeof(unsigned int));
	if (m == NULL) {
		throw("Failed to allocate memory for uint matrix.\n", MemoryError);
	}
	memset(m, 0, r * c * sizeof(unsigned int));
	return m;
}

/**
 * @brief Safely free float matrix data
 */
void free_mat_float(float *m) {
	if (m != NULL) {  // Good practice to check before free
		free(m);
	}
}

/**
 * @brief Safely free uint matrix data
 */
void free_mat_uint(unsigned int *m) {
	if (m != NULL) {  // Good practice to check before free
		free(m);
	}
}