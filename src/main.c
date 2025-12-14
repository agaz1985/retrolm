#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "matrix.h"

int main() {
	logger("Allocate memory...\n", DEBUG);

	struct Matrix2D m1 = new_mat(2, 3);
	struct Matrix2D m2 = new_mat(3, 2);
	struct Matrix2D m3 = new_mat(2, 2);

	/* m1: 2 rows, 3 cols */
	*mat_at(&m1, 0, 0) = 1;
	*mat_at(&m1, 0, 1) = 2;
	*mat_at(&m1, 0, 2) = 3;
	*mat_at(&m1, 1, 0) = 4;
	*mat_at(&m1, 1, 1) = 5;
	*mat_at(&m1, 1, 2) = 6;

	/* m2: 3 rows, 2 cols */
	*mat_at(&m2, 0, 0) = 1;
	*mat_at(&m2, 0, 1) = 2;
	*mat_at(&m2, 1, 0) = 3;
	*mat_at(&m2, 1, 1) = 4;
	*mat_at(&m2, 2, 0) = 5;
	*mat_at(&m2, 2, 1) = 6;

	logger("Matrix multiplication...\n", INFO);
	matmul(&m1, &m2, &m3);

	logger("Result:\n", DEBUG);
	print_mat(&m3);

	logger("Deallocate memory...\n", DEBUG);
	del_mat(&m1);
	del_mat(&m2);
	del_mat(&m3);

	return 0;
}