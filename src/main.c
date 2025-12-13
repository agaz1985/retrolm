#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "matrix.h"
#include "memory.h"
#include "exceptions.h"


extern long add_numbers(long a, long b);


void print_mat(struct Matrix2D m) {
	char buffer[4096] = "\n";
	int offset = 1;
	
	for (unsigned int i = 0; i < m.r; ++i) {
		for (unsigned int j = 0; j < m.c; ++j) {
			offset += sprintf(buffer + offset, "%f,", m.data[i][j]);
		}
		offset += sprintf(buffer + offset, "\n");
	}
	
	logger(buffer, INFO);
}

void matmul(struct Matrix2D m1, struct Matrix2D m2, struct Matrix2D out) {
   if(m1.c != m2.r) {
   	throw("Matrix dimensions do not match!\n", InvalidInput);
   }

   for (unsigned int x = 0; x < m1.r; ++x) {
   	for (unsigned int z = 0; z < m2.c; ++z) {
   	  for (unsigned int y = 0; y < m1.c; ++y) {
   		out.data[x][z] += m1.data[x][y] * m2.data[y][z];
   	  }
    }
   }
}

int main() {
	long result = add_numbers(5, 3);
	printf("The result is: %ld\n", result);

	logger("Allocate memory...\n", DEBUG);

	struct Matrix2D m1 = new_mat(2, 3);
	struct Matrix2D m2 = new_mat(3, 2);
	struct Matrix2D m3 = new_mat(2, 2);

	m1.data[0][0] = 1; m1.data[0][1] = 2; m1.data[0][2] = 3;
	m1.data[1][0] = 4; m1.data[1][1] = 5; m1.data[1][2] = 6;

	m2.data[0][0] = 1; m2.data[0][1] = 2;
	m2.data[1][0] = 3; m2.data[1][1] = 4;
	m2.data[2][0] = 5; m2.data[2][1] = 6;

	logger("Matrix multiplication...\n", INFO);
	matmul(m1, m2, m3);

	logger("Result:\n", DEBUG);
	print_mat(m3);

	logger("Deallocate memory...\n", DEBUG);
	del_mat(m1);
	del_mat(m2);
	del_mat(m3);

	return 0;
}