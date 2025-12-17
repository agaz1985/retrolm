#ifndef _RLM_MEMORY_H
#define _RLM_MEMORY_H

float* alloc_mat_float(unsigned int r, unsigned int c);
unsigned int* alloc_mat_uint(unsigned int r, unsigned int c);
void free_mat_float(float *m);
void free_mat_uint(unsigned int *m);

#endif // _RLM_MEMORY_H