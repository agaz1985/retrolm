/**
 * @file loader.c
 * @brief Implementation of model weight loading from binary files
 */

#include "loader.h"
#include "logger.h"
#include "exceptions.h"

#include <stdio.h>

/**
 * @brief Load a single weight matrix from binary file
 * 
 * Reads matrix dimensions (rows, cols) and data from a binary file.
 * Format: [uint32 rows][uint32 cols][float32 data...]
 */
struct Matrix2D load_weight_matrix(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        char msg[256];
        sprintf(msg, "Failed to open weight file: %s\n", filepath);
        throw(msg, FileError);
    }
    
    unsigned int rows, cols;
    if (fread(&rows, sizeof(unsigned int), 1, f) != 1) {
        fclose(f);
        throw("Failed to read matrix rows\n", FileError);
    }
    if (fread(&cols, sizeof(unsigned int), 1, f) != 1) {
        fclose(f);
        throw("Failed to read matrix cols\n", FileError);
    }
    
    struct Matrix2D m = mat_new(rows, cols);
    size_t expected = rows * cols;
    if (fread(m.data, sizeof(float), expected, f) != expected) {
        fclose(f);
        mat_free(&m);
        throw("Failed to read matrix data\n", FileError);
    }
    
    fclose(f);
    return m;
}

/**
 * @brief Load all transformer model weights from directory
 * 
 * Sequentially loads all weight matrices for the transformer model.
 * Implements weight tying by copying token embeddings to LM head weights.
 * 
 * The function logs each weight file as it's loaded for debugging.
 */
struct TransformerParameters load_model_weights(const char *weights_dir) {
    char filepath[256];
    struct TransformerParameters p;
    
    logger("Loading model weights...\n", INFO);
    
    // Token embeddings
    sprintf(filepath, "%s/token_embed.bin", weights_dir);
    logger("Loading token_embed.bin\n", DEBUG);
    p.token_embed.weight_matrix = load_weight_matrix(filepath);
    
    // Positional embeddings
    sprintf(filepath, "%s/pos_embed.bin", weights_dir);
    logger("Loading pos_embed.bin\n", DEBUG);
    p.pos_embed = load_weight_matrix(filepath);
    
    // Attention Q
    sprintf(filepath, "%s/Wq_weight.bin", weights_dir);
    logger("Loading Wq_weight.bin\n", DEBUG);
    p.attn.Wq.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/Wq_bias.bin", weights_dir);
    logger("Loading Wq_bias.bin\n", DEBUG);
    p.attn.Wq.bias = load_weight_matrix(filepath);
    
    // Attention K
    sprintf(filepath, "%s/Wk_weight.bin", weights_dir);
    logger("Loading Wk_weight.bin\n", DEBUG);
    p.attn.Wk.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/Wk_bias.bin", weights_dir);
    logger("Loading Wk_bias.bin\n", DEBUG);
    p.attn.Wk.bias = load_weight_matrix(filepath);
    
    // Attention V
    sprintf(filepath, "%s/Wv_weight.bin", weights_dir);
    logger("Loading Wv_weight.bin\n", DEBUG);
    p.attn.Wv.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/Wv_bias.bin", weights_dir);
    logger("Loading Wv_bias.bin\n", DEBUG);
    p.attn.Wv.bias = load_weight_matrix(filepath);
    
    // Attention O
    sprintf(filepath, "%s/Wo_weight.bin", weights_dir);
    logger("Loading Wo_weight.bin\n", DEBUG);
    p.attn.Wo.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/Wo_bias.bin", weights_dir);
    logger("Loading Wo_bias.bin\n", DEBUG);
    p.attn.Wo.bias = load_weight_matrix(filepath);
    
    // Feed-forward W1
    sprintf(filepath, "%s/W1_weight.bin", weights_dir);
    logger("Loading W1_weight.bin\n", DEBUG);
    p.W1.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/W1_bias.bin", weights_dir);
    logger("Loading W1_bias.bin\n", DEBUG);
    p.W1.bias = load_weight_matrix(filepath);
    
    // Feed-forward W2
    sprintf(filepath, "%s/W2_weight.bin", weights_dir);
    logger("Loading W2_weight.bin\n", DEBUG);
    p.W2.weights = load_weight_matrix(filepath);
    sprintf(filepath, "%s/W2_bias.bin", weights_dir);
    logger("Loading W2_bias.bin\n", DEBUG);
    p.W2.bias = load_weight_matrix(filepath);
    
    // LM head (bias only, weights are tied to token_embed)
    sprintf(filepath, "%s/lm_head_bias.bin", weights_dir);
    logger("Loading lm_head_bias.bin\n", DEBUG);
    p.lm_head.bias = load_weight_matrix(filepath);
    
    // Weight tying
    p.lm_head.weights = mat_copy(&p.token_embed.weight_matrix);
    
    logger("✓ All weights loaded successfully!\n", INFO);
    return p;
}