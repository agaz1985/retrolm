/**
 * @file test_utils.c
 * @brief Unit tests for utility functions
 */

#include <stdio.h>
#include <stdlib.h>
#include "test_framework.h"
#include "../src/utils.h"

int test_print_retrolm() {
    /* Test that print_retrolm doesn't crash */
    /* Since it only prints to stdout, we mainly verify it executes */
    print_retrolm();
    
    ASSERT(1, "print_retrolm should execute without crashing");
    return 1;
}

void run_utils_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Utils Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_print_retrolm);
}
