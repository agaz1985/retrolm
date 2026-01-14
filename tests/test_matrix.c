/**
 * @file test_matrix.c
 * @brief Unit tests for matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include "test_framework.h"
#include "../src/matrix.h"
#include "../src/exceptions.h"

/* Exception handling for testing */
extern jmp_buf exception_env;
extern int exception_pending;

int test_matrix_creation() {
    struct Matrix2D m = mat_new(3, 4);
    
    ASSERT(m.r == 3, "Matrix should have 3 rows");
    ASSERT(m.c == 4, "Matrix should have 4 columns");
    ASSERT(m.data != NULL, "Matrix data should not be NULL");
    
    /* Check zero initialization */
    for (unsigned int i = 0; i < m.r * m.c; i++) {
        ASSERT_FLOAT_EQ(m.data[i], 0.0f, "Matrix should be zero-initialized");
    }
    
    mat_free(&m);
    ASSERT(m.data == NULL, "Matrix data should be NULL after free");
    ASSERT(m.r == 0, "Matrix rows should be 0 after free");
    ASSERT(m.c == 0, "Matrix columns should be 0 after free");
    
    return 1;
}

int test_matrix_uint_creation() {
    struct Matrix2D_UInt m = mat_uint_new(2, 3);
    
    ASSERT(m.r == 2, "Matrix should have 2 rows");
    ASSERT(m.c == 3, "Matrix should have 3 columns");
    ASSERT(m.data != NULL, "Matrix data should not be NULL");
    
    /* Check zero initialization */
    for (unsigned int i = 0; i < m.r * m.c; i++) {
        ASSERT_INT_EQ(m.data[i], 0, "Matrix should be zero-initialized");
    }
    
    mat_uint_free(&m);
    return 1;
}

int test_matrix_indices() {
    struct Matrix2D_UInt indices = indices_new(5);
    
    ASSERT(indices.r == 1, "Indices should have 1 row");
    ASSERT(indices.c == 5, "Indices should have 5 columns");
    
    for (unsigned int i = 0; i < 5; i++) {
        ASSERT_INT_EQ(indices.data[i], i, "Indices should be sequential");
    }
    
    mat_uint_free(&indices);
    return 1;
}

int test_matrix_at() {
    struct Matrix2D m = mat_new(3, 3);
    
    /* Set some values */
    *mat_at(&m, 0, 0) = 1.0f;
    *mat_at(&m, 1, 1) = 2.0f;
    *mat_at(&m, 2, 2) = 3.0f;
    
    /* Check values */
    ASSERT_FLOAT_EQ(*mat_at(&m, 0, 0), 1.0f, "Element (0,0) should be 1.0");
    ASSERT_FLOAT_EQ(*mat_at(&m, 1, 1), 2.0f, "Element (1,1) should be 2.0");
    ASSERT_FLOAT_EQ(*mat_at(&m, 2, 2), 3.0f, "Element (2,2) should be 3.0");
    
    mat_free(&m);
    return 1;
}

int test_matrix_multiplication() {
    /* Create 2x3 matrix */
    struct Matrix2D m1 = mat_new(2, 3);
    m1.data[0] = 1.0f; m1.data[1] = 2.0f; m1.data[2] = 3.0f;
    m1.data[3] = 4.0f; m1.data[4] = 5.0f; m1.data[5] = 6.0f;
    
    /* Create 3x2 matrix */
    struct Matrix2D m2 = mat_new(3, 2);
    m2.data[0] = 7.0f;  m2.data[1] = 8.0f;
    m2.data[2] = 9.0f;  m2.data[3] = 10.0f;
    m2.data[4] = 11.0f; m2.data[5] = 12.0f;
    
    /* Multiply: result should be 2x2 */
    struct Matrix2D result = mat_mul(&m1, &m2);
    
    ASSERT(result.r == 2, "Result should have 2 rows");
    ASSERT(result.c == 2, "Result should have 2 columns");
    
    /* Expected values:
     * [1*7 + 2*9 + 3*11,   1*8 + 2*10 + 3*12]   [58,  64]
     * [4*7 + 5*9 + 6*11,   4*8 + 5*10 + 6*12] = [139, 154]
     */
    ASSERT_FLOAT_EQ(result.data[0], 58.0f, "Result[0,0] should be 58");
    ASSERT_FLOAT_EQ(result.data[1], 64.0f, "Result[0,1] should be 64");
    ASSERT_FLOAT_EQ(result.data[2], 139.0f, "Result[1,0] should be 139");
    ASSERT_FLOAT_EQ(result.data[3], 154.0f, "Result[1,1] should be 154");
    
    mat_free(&m1);
    mat_free(&m2);
    mat_free(&result);
    return 1;
}

int test_matrix_addition() {
    struct Matrix2D m1 = mat_new(2, 2);
    m1.data[0] = 1.0f; m1.data[1] = 2.0f;
    m1.data[2] = 3.0f; m1.data[3] = 4.0f;
    
    struct Matrix2D m2 = mat_new(2, 2);
    m2.data[0] = 5.0f; m2.data[1] = 6.0f;
    m2.data[2] = 7.0f; m2.data[3] = 8.0f;
    
    struct Matrix2D result = mat_add(&m1, &m2);
    
    ASSERT_FLOAT_EQ(result.data[0], 6.0f, "Result[0,0] should be 6");
    ASSERT_FLOAT_EQ(result.data[1], 8.0f, "Result[0,1] should be 8");
    ASSERT_FLOAT_EQ(result.data[2], 10.0f, "Result[1,0] should be 10");
    ASSERT_FLOAT_EQ(result.data[3], 12.0f, "Result[1,1] should be 12");
    
    mat_free(&m1);
    mat_free(&m2);
    mat_free(&result);
    return 1;
}

int test_matrix_subtraction() {
    struct Matrix2D m1 = mat_new(2, 2);
    m1.data[0] = 10.0f; m1.data[1] = 20.0f;
    m1.data[2] = 30.0f; m1.data[3] = 40.0f;
    
    struct Matrix2D m2 = mat_new(2, 2);
    m2.data[0] = 1.0f; m2.data[1] = 2.0f;
    m2.data[2] = 3.0f; m2.data[3] = 4.0f;
    
    struct Matrix2D result = mat_sub(&m1, &m2);
    
    ASSERT_FLOAT_EQ(result.data[0], 9.0f, "Result[0,0] should be 9");
    ASSERT_FLOAT_EQ(result.data[1], 18.0f, "Result[0,1] should be 18");
    ASSERT_FLOAT_EQ(result.data[2], 27.0f, "Result[1,0] should be 27");
    ASSERT_FLOAT_EQ(result.data[3], 36.0f, "Result[1,1] should be 36");
    
    mat_free(&m1);
    mat_free(&m2);
    mat_free(&result);
    return 1;
}

int test_matrix_transpose() {
    struct Matrix2D m = mat_new(2, 3);
    m.data[0] = 1.0f; m.data[1] = 2.0f; m.data[2] = 3.0f;
    m.data[3] = 4.0f; m.data[4] = 5.0f; m.data[5] = 6.0f;
    
    struct Matrix2D result = mat_transpose(&m);
    
    ASSERT(result.r == 3, "Transposed matrix should have 3 rows");
    ASSERT(result.c == 2, "Transposed matrix should have 2 columns");
    
    ASSERT_FLOAT_EQ(result.data[0], 1.0f, "Result[0,0] should be 1");
    ASSERT_FLOAT_EQ(result.data[1], 4.0f, "Result[0,1] should be 4");
    ASSERT_FLOAT_EQ(result.data[2], 2.0f, "Result[1,0] should be 2");
    ASSERT_FLOAT_EQ(result.data[3], 5.0f, "Result[1,1] should be 5");
    ASSERT_FLOAT_EQ(result.data[4], 3.0f, "Result[2,0] should be 3");
    ASSERT_FLOAT_EQ(result.data[5], 6.0f, "Result[2,1] should be 6");
    
    mat_free(&m);
    mat_free(&result);
    return 1;
}

int test_matrix_copy() {
    struct Matrix2D m = mat_new(2, 2);
    m.data[0] = 1.0f; m.data[1] = 2.0f;
    m.data[2] = 3.0f; m.data[3] = 4.0f;
    
    struct Matrix2D copy = mat_copy(&m);
    
    ASSERT(copy.r == m.r, "Copy should have same rows");
    ASSERT(copy.c == m.c, "Copy should have same columns");
    
    for (unsigned int i = 0; i < m.r * m.c; i++) {
        ASSERT_FLOAT_EQ(copy.data[i], m.data[i], "Copy should have same values");
    }
    
    /* Modify original, copy should be unchanged */
    m.data[0] = 99.0f;
    ASSERT_FLOAT_EQ(copy.data[0], 1.0f, "Copy should be independent");
    
    mat_free(&m);
    mat_free(&copy);
    return 1;
}

void run_matrix_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Matrix Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_matrix_creation);
    RUN_TEST(test_matrix_uint_creation);
    RUN_TEST(test_matrix_indices);
    RUN_TEST(test_matrix_at);
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_matrix_addition);
    RUN_TEST(test_matrix_subtraction);
    RUN_TEST(test_matrix_transpose);
    RUN_TEST(test_matrix_copy);
}
