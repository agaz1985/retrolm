/**
 * @file test_framework.h
 * @brief Minimal testing framework for RetroLM unit tests
 */

#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Test statistics - declared extern, defined in test_runner.c */
extern int tests_run;
extern int tests_passed;
extern int tests_failed;

/* Color codes for output */
#define COLOR_GREEN "\033[0;32m"
#define COLOR_RED "\033[0;31m"
#define COLOR_YELLOW "\033[0;33m"
#define COLOR_BLUE "\033[0;34m"
#define COLOR_RESET "\033[0m"

/* Epsilon for floating-point comparisons */
#define EPSILON 1e-5f

/**
 * @brief Assert that a condition is true
 */
#define ASSERT(condition, message) do { \
    tests_run++; \
    if (!(condition)) { \
        printf(COLOR_RED "✗ FAIL: %s\n" COLOR_RESET, message); \
        printf("  Line %d in %s\n", __LINE__, __FILE__); \
        tests_failed++; \
        return 0; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/**
 * @brief Assert that two floats are equal within epsilon
 */
#define ASSERT_FLOAT_EQ(actual, expected, message) do { \
    tests_run++; \
    float diff = fabsf((actual) - (expected)); \
    if (diff > EPSILON) { \
        printf(COLOR_RED "✗ FAIL: %s\n" COLOR_RESET, message); \
        printf("  Expected: %f, Got: %f (diff: %f)\n", (float)(expected), (float)(actual), diff); \
        printf("  Line %d in %s\n", __LINE__, __FILE__); \
        tests_failed++; \
        return 0; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/**
 * @brief Assert that two integers are equal
 */
#define ASSERT_INT_EQ(actual, expected, message) do { \
    tests_run++; \
    if ((actual) != (expected)) { \
        printf(COLOR_RED "✗ FAIL: %s\n" COLOR_RESET, message); \
        printf("  Expected: %d, Got: %d\n", (int)(expected), (int)(actual)); \
        printf("  Line %d in %s\n", __LINE__, __FILE__); \
        tests_failed++; \
        return 0; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/**
 * @brief Run a test function
 */
#define RUN_TEST(test_func) do { \
    printf(COLOR_BLUE "Running %s..." COLOR_RESET, #test_func); \
    if (test_func()) { \
        printf(COLOR_GREEN " ✓ PASS\n" COLOR_RESET); \
    } else { \
        printf(COLOR_RED " ✗ FAILED\n" COLOR_RESET); \
    } \
} while(0)

/**
 * @brief Print test summary
 */
#define PRINT_TEST_SUMMARY() do { \
    printf("\n" COLOR_BLUE "==========================================\n"); \
    printf("Test Summary\n"); \
    printf("==========================================" COLOR_RESET "\n"); \
    printf("Total assertions: %d\n", tests_run); \
    printf(COLOR_GREEN "Passed: %d\n" COLOR_RESET, tests_passed); \
    if (tests_failed > 0) { \
        printf(COLOR_RED "Failed: %d\n" COLOR_RESET, tests_failed); \
    } else { \
        printf("Failed: 0\n"); \
    } \
    float pass_rate = tests_run > 0 ? (100.0f * tests_passed / tests_run) : 0.0f; \
    printf("Pass rate: %.1f%%\n", pass_rate); \
    printf("\n"); \
} while(0)

#endif // TEST_FRAMEWORK_H
