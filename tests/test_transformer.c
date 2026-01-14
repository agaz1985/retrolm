/**
 * @file test_transformer.c
 * @brief Unit tests for transformer model
 */

#include <stdio.h>
#include <stdlib.h>
#include "test_framework.h"
#include "../src/transformer.h"
#include "../src/matrix.h"

int test_transformer_new() {
    /* Create a small transformer: seq_len=8, embed_dim=16, ff_dim=32, vocab_size=100 */
    struct TransformerParameters model = transformer_new(8, 16, 32, 100);
    
    /* Check token embeddings */
    ASSERT(model.token_embed.weight_matrix.r == 100, "Token embed should have vocab_size rows");
    ASSERT(model.token_embed.weight_matrix.c == 16, "Token embed should have embed_dim columns");
    
    /* Check positional embeddings */
    ASSERT(model.pos_embed.r == 8, "Pos embed should have seq_len rows");
    ASSERT(model.pos_embed.c == 16, "Pos embed should have embed_dim columns");
    
    /* Check attention */
    ASSERT(model.attn.Wq.weights.r == 16, "Attention query should be embed_dim");
    
    /* Check feed-forward */
    ASSERT(model.W1.weights.r == 32, "W1 output should be ff_dim");
    ASSERT(model.W1.weights.c == 16, "W1 input should be embed_dim");
    ASSERT(model.W2.weights.r == 16, "W2 output should be embed_dim");
    ASSERT(model.W2.weights.c == 32, "W2 input should be ff_dim");
    
    /* Check LM head */
    ASSERT(model.lm_head.weights.r == 100, "LM head should output vocab_size");
    ASSERT(model.lm_head.weights.c == 16, "LM head input should be embed_dim");
    
    transformer_free(&model);
    return 1;
}

int test_transformer_forward_shape() {
    /* Create a tiny transformer for testing */
    struct TransformerParameters model = transformer_new(4, 8, 16, 20);
    
    /* Create input token sequence [1, 3, 7, 2] */
    struct Matrix2D_UInt input = mat_uint_new(1, 4);
    input.data[0] = 1;
    input.data[1] = 3;
    input.data[2] = 7;
    input.data[3] = 2;
    
    /* Forward pass with cache */
    struct AttentionCache cache = attention_cache_new(model.attn.Wq.weights.r);
    struct Matrix2D output = transformer_forward(&input, &model, &cache, 0);
    
    /* Check output shape */
    ASSERT(output.r == 4, "Output should have seq_len rows");
    ASSERT(output.c == 20, "Output should have vocab_size columns");
    
    /* Check that output data is allocated */
    ASSERT(output.data != NULL, "Output data should be allocated");
    
    attention_cache_free(&cache);
    mat_uint_free(&input);
    mat_free(&output);
    transformer_free(&model);
    return 1;
}

void run_transformer_tests() {
    printf("\n" COLOR_YELLOW "======================================\n");
    printf("Running Transformer Tests\n");
    printf("======================================" COLOR_RESET "\n\n");
    
    RUN_TEST(test_transformer_new);
    RUN_TEST(test_transformer_forward_shape);
}
