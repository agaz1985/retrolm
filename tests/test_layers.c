/**
 * @file test_layers.c
 * @brief Unit tests for neural network layers
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/layers.h"
#include "../src/matrix.h"

/* ========================================
 * Linear Layer Tests
 * ======================================== */

int test_linear_new() {
    struct LinearParameters linear = linear_new(10, 5);
    
    ASSERT(linear.weights.r == 5, "Weight rows should match out_features");
    ASSERT(linear.weights.c == 10, "Weight cols should match in_features");
    ASSERT(linear.bias.r == 1, "Bias should have 1 row");
    ASSERT(linear.bias.c == 5, "Bias cols should match out_features");
    ASSERT(linear.weights.data != NULL, "Weights should be allocated");
    ASSERT(linear.bias.data != NULL, "Bias should be allocated");
    
    linear_free(&linear);
    return 1;
}

int test_linear_forward() {
    /* Create a simple 2x3 linear layer */
    struct LinearParameters linear = linear_new(2, 3);
    
    /* Set simple weights and bias */
    linear.weights.data[0] = 1.0f; linear.weights.data[1] = 0.0f;  /* Row 0 */
    linear.weights.data[2] = 0.0f; linear.weights.data[3] = 1.0f;  /* Row 1 */
    linear.weights.data[4] = 1.0f; linear.weights.data[5] = 1.0f;  /* Row 2 */
    
    linear.bias.data[0] = 0.1f;
    linear.bias.data[1] = 0.2f;
    linear.bias.data[2] = 0.3f;
    
    /* Input: [1, 2] */
    struct Matrix2D input = mat_new(1, 2);
    input.data[0] = 1.0f;
    input.data[1] = 2.0f;
    
    struct Matrix2D output = linear_forward(&input, &linear);
    
    ASSERT(output.r == 1, "Output should have 1 row");
    ASSERT(output.c == 3, "Output should have 3 columns");
    
    /* Expected: [1*1 + 2*0 + 0.1, 1*0 + 2*1 + 0.2, 1*1 + 2*1 + 0.3] 
     *         = [1.1, 2.2, 3.3] */
    ASSERT_FLOAT_EQ(output.data[0], 1.1f, "Output[0] should be 1.1");
    ASSERT_FLOAT_EQ(output.data[1], 2.2f, "Output[1] should be 2.2");
    ASSERT_FLOAT_EQ(output.data[2], 3.3f, "Output[2] should be 3.3");
    
    mat_free(&input);
    mat_free(&output);
    linear_free(&linear);
    return 1;
}

/* ========================================
 * Embeddings Tests
 * ======================================== */

int test_embeddings_new() {
    struct EmbeddingsParameters embed = embeddings_new(100, 64);
    
    ASSERT(embed.weight_matrix.r == 100, "Embeddings should have vocab_size rows");
    ASSERT(embed.weight_matrix.c == 64, "Embeddings should have embed_dim columns");
    ASSERT(embed.weight_matrix.data != NULL, "Embedding matrix should be allocated");
    
    embeddings_free(&embed);
    return 1;
}

int test_embeddings_forward() {
    /* Create embeddings with vocab=4, dim=3 */
    struct EmbeddingsParameters embed = embeddings_new(4, 3);
    
    /* Set embedding vectors */
    embed.weight_matrix.data[0] = 1.0f; embed.weight_matrix.data[1] = 0.0f; embed.weight_matrix.data[2] = 0.0f;
    embed.weight_matrix.data[3] = 0.0f; embed.weight_matrix.data[4] = 1.0f; embed.weight_matrix.data[5] = 0.0f;
    embed.weight_matrix.data[6] = 0.0f; embed.weight_matrix.data[7] = 0.0f; embed.weight_matrix.data[8] = 1.0f;
    embed.weight_matrix.data[9] = 1.0f; embed.weight_matrix.data[10] = 1.0f; embed.weight_matrix.data[11] = 1.0f;
    
    /* Token indices: [0, 2, 3] */
    struct Matrix2D_UInt indices = mat_uint_new(1, 3);
    indices.data[0] = 0;
    indices.data[1] = 2;
    indices.data[2] = 3;
    
    struct Matrix2D output = embeddings_forward(&indices, &embed);
    
    ASSERT(output.r == 3, "Output should have same length as input sequence");
    ASSERT(output.c == 3, "Output should have embedding dimension");
    
    /* Check that embeddings are correctly selected */
    ASSERT_FLOAT_EQ(output.data[0], 1.0f, "First embedding should be [1,0,0]");
    ASSERT_FLOAT_EQ(output.data[1], 0.0f, "First embedding should be [1,0,0]");
    ASSERT_FLOAT_EQ(output.data[2], 0.0f, "First embedding should be [1,0,0]");
    
    ASSERT_FLOAT_EQ(output.data[3], 0.0f, "Second embedding should be [0,0,1]");
    ASSERT_FLOAT_EQ(output.data[4], 0.0f, "Second embedding should be [0,0,1]");
    ASSERT_FLOAT_EQ(output.data[5], 1.0f, "Second embedding should be [0,0,1]");
    
    mat_uint_free(&indices);
    mat_free(&output);
    embeddings_free(&embed);
    return 1;
}

/* ========================================
 * Self-Attention Tests
 * ======================================== */

int test_self_attention_new() {
    struct SelfAttentionParameters attn = attention_new(64);
    
    ASSERT(attn.Wq.weights.r == 64, "Query weights should be square");
    ASSERT(attn.Wk.weights.r == 64, "Key weights should be square");
    ASSERT(attn.Wv.weights.r == 64, "Value weights should be square");
    ASSERT(attn.Wo.weights.r == 64, "Output weights should be square");
    
    attention_free(&attn);
    return 1;
}

void run_layers_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Layers Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_linear_new);
    RUN_TEST(test_linear_forward);
    RUN_TEST(test_embeddings_new);
    RUN_TEST(test_embeddings_forward);
    RUN_TEST(test_self_attention_new);
}
