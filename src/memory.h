/**
 * @file memory.h
 * @brief Low-level memory allocation for matrix data
 * 
 * Provides memory allocation and deallocation wrappers for matrix data arrays.
 * These functions handle error checking and zero-initialization of allocated memory.
 */

#ifndef _RLM_MEMORY_H
#define _RLM_MEMORY_H

/**
 * @brief Allocate and zero-initialize a float array for matrix data
 * 
 * @param r Number of rows
 * @param c Number of columns
 * @return Pointer to allocated and zero-initialized float array of size r*c
 * 
 * @throws MemoryError if allocation fails
 * @note Memory is zero-initialized using memset
 */
float* alloc_mat_float(unsigned int r, unsigned int c);

/**
 * @brief Allocate and zero-initialize an unsigned int array for matrix data
 * 
 * @param r Number of rows
 * @param c Number of columns
 * @return Pointer to allocated and zero-initialized uint array of size r*c
 * 
 * @throws MemoryError if allocation fails
 * @note Memory is zero-initialized using memset
 */
unsigned int* alloc_mat_uint(unsigned int r, unsigned int c);

/**
 * @brief Free float matrix data
 * 
 * @param m Pointer to float array to free (can be NULL)
 * 
 * @note Safe to call with NULL pointer
 */
void free_mat_float(float *m);

/**
 * @brief Free unsigned int matrix data
 * 
 * @param m Pointer to uint array to free (can be NULL)
 * 
 * @note Safe to call with NULL pointer
 */
void free_mat_uint(unsigned int *m);

#endif // _RLM_MEMORY_H