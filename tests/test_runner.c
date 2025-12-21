/**
 * @file test_runner.c
 * @brief Main test runner for RetroLM unit tests
 */

#include <stdio.h>
#include <stdlib.h>
#include "test_framework.h"

/* Define test statistics globals */
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

/* External test suite functions */
extern void run_matrix_tests();
extern void run_matrix_ops_tests();
extern void run_activations_tests();
extern void run_memory_tests();
extern void run_utils_tests();
extern void run_sampling_tests();
extern void run_layers_tests();
extern void run_transformer_tests();

int main(int argc, char *argv[]) {
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║       RetroLM Unit Test Suite         ║\n");
    printf("╚════════════════════════════════════════╝\n");
    
    /* Run all test suites */
    run_memory_tests();
    run_matrix_tests();
    run_matrix_ops_tests();
    run_activations_tests();
    run_utils_tests();
    run_sampling_tests();
    run_layers_tests();
    run_transformer_tests();
    
    /* Print summary */
    PRINT_TEST_SUMMARY();
    
    /* Return exit code based on test results */
    if (tests_failed > 0) {
        printf(COLOR_RED "Some tests failed!\n" COLOR_RESET);
        return 1;
    } else {
        printf(COLOR_GREEN "All tests passed!\n" COLOR_RESET);
        return 0;
    }
}
