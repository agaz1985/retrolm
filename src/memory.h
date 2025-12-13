#ifndef _RLM_MEMORY_H
#define _RLM_MEMORY_H

float** alloc_mat(unsigned int r, unsigned int c);
void free_mat(float** m, unsigned int r, unsigned int c);

#endif // _RLM_MEMORY_H