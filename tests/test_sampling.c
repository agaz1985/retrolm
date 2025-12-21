/**
 * @file test_sampling.c
 * @brief Unit tests for sampling functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test_framework.h"
#include "../src/sampling.h"

int test_sample_from_logits_basic() {
    /* Test basic sampling with simple logits */
    float logits[3] = {1.0f, 2.0f, 3.0f};
    
    srand(42);  /* Fixed seed for reproducibility */
    
    /* Sample multiple times */
    for (int i = 0; i < 10; i++) {
        unsigned int token = sample_from_logits(logits, 3, 1.0f);
        ASSERT(token < 3, "Sampled token should be within vocabulary");
    }
    
    return 1;
}

int test_sample_from_logits_temperature() {
    /* Test temperature effects on sampling */
    float logits[5] = {0.0f, 0.0f, 10.0f, 0.0f, 0.0f};
    
    srand(123);
    
    /* With very low temperature, should strongly prefer highest logit */
    unsigned int token_low_temp = sample_from_logits(logits, 5, 0.1f);
    ASSERT(token_low_temp == 2, "Low temperature should prefer highest logit");
    
    /* Reset logits */
    logits[0] = 0.0f; logits[1] = 0.0f; logits[2] = 10.0f; 
    logits[3] = 0.0f; logits[4] = 0.0f;
    
    /* With high temperature, distribution is more uniform */
    token_low_temp = sample_from_logits(logits, 5, 2.0f);
    ASSERT(token_low_temp < 5, "High temperature sample should be valid");
    
    return 1;
}

int test_sample_from_logits_uniform() {
    /* Test with uniform logits */
    float logits[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    srand(time(NULL));
    
    /* Sample should be valid */
    unsigned int token = sample_from_logits(logits, 4, 1.0f);
    ASSERT(token < 4, "Sample from uniform distribution should be valid");
    
    return 1;
}

void run_sampling_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Sampling Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_sample_from_logits_basic);
    RUN_TEST(test_sample_from_logits_temperature);
    RUN_TEST(test_sample_from_logits_uniform);
}
