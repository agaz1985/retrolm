/**
 * @file test_memory.c
 * @brief Unit tests for memory allocation functions
 */

#include <stdio.h>
#include <stdlib.h>
#include "test_framework.h"
#include "../src/memory.h"

int test_alloc_mat_float() {
    float *data = alloc_mat_float(3, 4);
    
    ASSERT(data != NULL, "Float matrix allocation should succeed");
    
    /* Check zero initialization */
    for (unsigned int i = 0; i < 12; i++) {
        ASSERT_FLOAT_EQ(data[i], 0.0f, "Float matrix should be zero-initialized");
    }
    
    /* Test that we can write to it */
    data[0] = 42.0f;
    ASSERT_FLOAT_EQ(data[0], 42.0f, "Should be able to write to allocated memory");
    
    free_mat_float(data);
    return 1;
}

int test_alloc_mat_uint() {
    unsigned int *data = alloc_mat_uint(2, 5);
    
    ASSERT(data != NULL, "Uint matrix allocation should succeed");
    
    /* Check zero initialization */
    for (unsigned int i = 0; i < 10; i++) {
        ASSERT_INT_EQ(data[i], 0, "Uint matrix should be zero-initialized");
    }
    
    /* Test that we can write to it */
    data[0] = 123;
    ASSERT_INT_EQ(data[0], 123, "Should be able to write to allocated memory");
    
    free_mat_uint(data);
    return 1;
}

int test_free_mat_float_null() {
    /* Should not crash on NULL pointer */
    free_mat_float(NULL);
    
    ASSERT(1, "free_mat_float(NULL) should not crash");
    return 1;
}

int test_free_mat_uint_null() {
    /* Should not crash on NULL pointer */
    free_mat_uint(NULL);
    
    ASSERT(1, "free_mat_uint(NULL) should not crash");
    return 1;
}

int test_alloc_large_matrix() {
    /* Test allocation of a reasonably large matrix */
    float *data = alloc_mat_float(100, 100);
    
    ASSERT(data != NULL, "Large matrix allocation should succeed");
    
    /* Spot check some values */
    ASSERT_FLOAT_EQ(data[0], 0.0f, "First element should be zero");
    ASSERT_FLOAT_EQ(data[9999], 0.0f, "Last element should be zero");
    
    /* Write some values */
    data[50 * 100 + 50] = 3.14159f; /* Middle element */
    ASSERT_FLOAT_EQ(data[5050], 3.14159f, "Should access middle element correctly");
    
    free_mat_float(data);
    return 1;
}

int test_alloc_single_element() {
    /* Test edge case: 1x1 matrix */
    float *data = alloc_mat_float(1, 1);
    
    ASSERT(data != NULL, "1x1 matrix allocation should succeed");
    ASSERT_FLOAT_EQ(data[0], 0.0f, "Single element should be zero");
    
    data[0] = 99.99f;
    ASSERT_FLOAT_EQ(data[0], 99.99f, "Should be able to modify single element");
    
    free_mat_float(data);
    return 1;
}

int test_multiple_allocations() {
    /* Test multiple allocations don't interfere */
    float *data1 = alloc_mat_float(2, 2);
    float *data2 = alloc_mat_float(3, 3);
    unsigned int *data3 = alloc_mat_uint(4, 4);
    
    ASSERT(data1 != NULL, "First allocation should succeed");
    ASSERT(data2 != NULL, "Second allocation should succeed");
    ASSERT(data3 != NULL, "Third allocation should succeed");
    
    /* Write to each */
    data1[0] = 1.0f;
    data2[0] = 2.0f;
    data3[0] = 3;
    
    /* Verify they don't interfere */
    ASSERT_FLOAT_EQ(data1[0], 1.0f, "First allocation should be independent");
    ASSERT_FLOAT_EQ(data2[0], 2.0f, "Second allocation should be independent");
    ASSERT_INT_EQ(data3[0], 3, "Third allocation should be independent");
    
    free_mat_float(data1);
    free_mat_float(data2);
    free_mat_uint(data3);
    
    return 1;
}

void run_memory_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Memory Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_alloc_mat_float);
    RUN_TEST(test_alloc_mat_uint);
    RUN_TEST(test_free_mat_float_null);
    RUN_TEST(test_free_mat_uint_null);
    RUN_TEST(test_alloc_large_matrix);
    RUN_TEST(test_alloc_single_element);
    RUN_TEST(test_multiple_allocations);
}
