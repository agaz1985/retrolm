/**
 * @file test_matrix_ops.c
 * @brief Unit tests for low-level matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_framework.h"
#include "../src/matrix_ops.h"
#include "../src/memory.h"

int test_matmul_basic() {
    /* Create 2x3 matrix */
    float m1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    /* Create 3x2 matrix */
    float m2[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    /* Result: 2x2 */
    float result[4];
    
    _matmul(m1, m2, result, 2, 3, 2);
    
    ASSERT_FLOAT_EQ(result[0], 58.0f, "Result[0,0] should be 58");
    ASSERT_FLOAT_EQ(result[1], 64.0f, "Result[0,1] should be 64");
    ASSERT_FLOAT_EQ(result[2], 139.0f, "Result[1,0] should be 139");
    ASSERT_FLOAT_EQ(result[3], 154.0f, "Result[1,1] should be 154");
    
    return 1;
}

int test_matadd_basic() {
    float m1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float m2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];
    
    _matadd(m1, m2, result, 2, 2);
    
    ASSERT_FLOAT_EQ(result[0], 6.0f, "Result[0] should be 6");
    ASSERT_FLOAT_EQ(result[1], 8.0f, "Result[1] should be 8");
    ASSERT_FLOAT_EQ(result[2], 10.0f, "Result[2] should be 10");
    ASSERT_FLOAT_EQ(result[3], 12.0f, "Result[3] should be 12");
    
    return 1;
}

int test_matadd_rowbroadcast() {
    /* Matrix 2x3 */
    float m1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    /* Row vector 1x3 */
    float m2[3] = {10.0f, 20.0f, 30.0f};
    float result[6];
    
    _matadd_rowbroadcast(m1, m2, result, 2, 3);
    
    ASSERT_FLOAT_EQ(result[0], 11.0f, "Result[0,0] should be 11");
    ASSERT_FLOAT_EQ(result[1], 22.0f, "Result[0,1] should be 22");
    ASSERT_FLOAT_EQ(result[2], 33.0f, "Result[0,2] should be 33");
    ASSERT_FLOAT_EQ(result[3], 14.0f, "Result[1,0] should be 14");
    ASSERT_FLOAT_EQ(result[4], 25.0f, "Result[1,1] should be 25");
    ASSERT_FLOAT_EQ(result[5], 36.0f, "Result[1,2] should be 36");
    
    return 1;
}

int test_matadd_colbroadcast() {
    /* Matrix 3x2 */
    float m1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    /* Column vector 3x1 */
    float m2[3] = {10.0f, 20.0f, 30.0f};
    float result[6];
    
    _matadd_colbroadcast(m1, m2, result, 3, 2);
    
    ASSERT_FLOAT_EQ(result[0], 11.0f, "Result[0,0] should be 11");
    ASSERT_FLOAT_EQ(result[1], 12.0f, "Result[0,1] should be 12");
    ASSERT_FLOAT_EQ(result[2], 23.0f, "Result[1,0] should be 23");
    ASSERT_FLOAT_EQ(result[3], 24.0f, "Result[1,1] should be 24");
    ASSERT_FLOAT_EQ(result[4], 35.0f, "Result[2,0] should be 35");
    ASSERT_FLOAT_EQ(result[5], 36.0f, "Result[2,1] should be 36");
    
    return 1;
}

int test_matsub_basic() {
    float m1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float m2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result[4];
    
    _matsub(m1, m2, result, 2, 2);
    
    ASSERT_FLOAT_EQ(result[0], 9.0f, "Result[0] should be 9");
    ASSERT_FLOAT_EQ(result[1], 18.0f, "Result[1] should be 18");
    ASSERT_FLOAT_EQ(result[2], 27.0f, "Result[2] should be 27");
    ASSERT_FLOAT_EQ(result[3], 36.0f, "Result[3] should be 36");
    
    return 1;
}

int test_matdiv_basic() {
    float m1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float m2[4] = {2.0f, 4.0f, 5.0f, 8.0f};
    float result[4];
    
    _matdiv(m1, m2, result, 2, 2);
    
    ASSERT_FLOAT_EQ(result[0], 5.0f, "Result[0] should be 5");
    ASSERT_FLOAT_EQ(result[1], 5.0f, "Result[1] should be 5");
    ASSERT_FLOAT_EQ(result[2], 6.0f, "Result[2] should be 6");
    ASSERT_FLOAT_EQ(result[3], 5.0f, "Result[3] should be 5");
    
    return 1;
}

int test_matdiv_rowbroadcast() {
    /* Matrix 2x3 */
    float m1[6] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    /* Row vector 1x3 - make sure dimensions are compatible */
    float m2[3] = {2.0f, 5.0f, 10.0f};
    float result[6];
    
    _matdiv_rowbroadcast(m1, m2, result, 2, 3);
    
    ASSERT_FLOAT_EQ(result[0], 5.0f, "Result[0,0] should be 5");
    ASSERT_FLOAT_EQ(result[1], 4.0f, "Result[0,1] should be 4");
    ASSERT_FLOAT_EQ(result[2], 3.0f, "Result[0,2] should be 3");
    ASSERT_FLOAT_EQ(result[3], 20.0f, "Result[1,0] should be 20");
    ASSERT_FLOAT_EQ(result[4], 10.0f, "Result[1,1] should be 10");
    ASSERT_FLOAT_EQ(result[5], 6.0f, "Result[1,2] should be 6");
    
    return 1;
}

int test_matscale() {
    float m[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    _matscale(m, 2, 2, 2.5f);
    
    ASSERT_FLOAT_EQ(m[0], 2.5f, "Result[0] should be 2.5");
    ASSERT_FLOAT_EQ(m[1], 5.0f, "Result[1] should be 5.0");
    ASSERT_FLOAT_EQ(m[2], 7.5f, "Result[2] should be 7.5");
    ASSERT_FLOAT_EQ(m[3], 10.0f, "Result[3] should be 10.0");
    
    return 1;
}

int test_matexp() {
    float m[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float result[4];
    
    _matexp(m, result, 2, 2);
    
    /* exp(0) = 1, exp(1) = 2.718..., exp(2) = 7.389..., exp(3) = 20.085... */
    ASSERT_FLOAT_EQ(result[0], 1.0f, "exp(0) should be 1");
    ASSERT(result[1] > 2.7f && result[1] < 2.8f, "exp(1) should be ~2.718");
    ASSERT(result[2] > 7.3f && result[2] < 7.4f, "exp(2) should be ~7.389");
    ASSERT(result[3] > 20.0f && result[3] < 20.1f, "exp(3) should be ~20.085");
    
    return 1;
}

int test_matsum_rowwise() {
    /* Matrix 2x3 */
    float m[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float result[2];
    
    _matsum_rowwise(m, result, 2, 3);
    
    ASSERT_FLOAT_EQ(result[0], 6.0f, "Row 0 sum should be 6");
    ASSERT_FLOAT_EQ(result[1], 15.0f, "Row 1 sum should be 15");
    
    return 1;
}

int test_matmax_rowwise() {
    /* Matrix 2x3 */
    float m[6] = {3.0f, 1.0f, 2.0f, 5.0f, 9.0f, 7.0f};
    float result[2];
    
    _matmax_rowwise(m, result, 2, 3);
    
    ASSERT_FLOAT_EQ(result[0], 3.0f, "Row 0 max should be 3");
    ASSERT_FLOAT_EQ(result[1], 9.0f, "Row 1 max should be 9");
    
    return 1;
}

void run_matrix_ops_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Matrix Operations Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_matmul_basic);
    RUN_TEST(test_matadd_basic);
    RUN_TEST(test_matadd_rowbroadcast);
    RUN_TEST(test_matadd_colbroadcast);
    RUN_TEST(test_matsub_basic);
    RUN_TEST(test_matdiv_basic);
    RUN_TEST(test_matdiv_rowbroadcast);
    RUN_TEST(test_matscale);
    RUN_TEST(test_matexp);
    RUN_TEST(test_matsum_rowwise);
    RUN_TEST(test_matmax_rowwise);
}
