#include "loader.h"
#include "logger.h"
#include "exceptions.h"

#include <stdio.h>
#include <string.h>

struct Matrix2D load_weight_matrix(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Failed to open weight file: %s\n", filepath);
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

static struct Matrix2D load_weight(const char *dir, const char *filename) {
    char filepath[1024];
    char logmsg[256];
    int written = snprintf(filepath, sizeof(filepath), "%s%s", dir, filename);
    if (written < 0 || (size_t)written >= sizeof(filepath)) {
        throw("Weight file path too long\n", ValueError);
    }
    snprintf(logmsg, sizeof(logmsg), "Loading %s\n", filename);
    logger(logmsg, DEBUG);
    return load_weight_matrix(filepath);
}

struct TransformerParameters load_model_weights(const char *weights_dir) {
    char weights_path[1024];
    const char *normalized_dir;
    struct TransformerParameters p;
    
    logger("Loading model weights...\n", INFO);

    // Validate and normalize weights_dir path
    size_t dir_len = strlen(weights_dir);
    if (dir_len == 0) {
        throw("Empty weights directory path\n", ValueError);
    }
    if (dir_len > 1000) {
        throw("Weights directory path too long\n", ValueError);
    }
    
    // Ensure path ends with separator
    char last_char = weights_dir[dir_len - 1];
#ifdef _WIN32
    if (last_char != '\\' && last_char != '/') {
        snprintf(weights_path, sizeof(weights_path), "%s\\", weights_dir);
        normalized_dir = weights_path;
    } else {
        normalized_dir = weights_dir;
    }
#else
    if (last_char != '/') {
        snprintf(weights_path, sizeof(weights_path), "%s/", weights_dir);
        normalized_dir = weights_path;
    } else {
        normalized_dir = weights_dir;
    }
#endif

    // Load embeddings
    p.token_embed.weight_matrix = load_weight(normalized_dir, "token_embed.bin");
    p.pos_embed = load_weight(normalized_dir, "pos_embed.bin");
    
    // Load attention weights
    p.attn.Wq.weights = load_weight(normalized_dir, "Wq_weight.bin");
    p.attn.Wq.bias = load_weight(normalized_dir, "Wq_bias.bin");
    p.attn.Wk.weights = load_weight(normalized_dir, "Wk_weight.bin");
    p.attn.Wk.bias = load_weight(normalized_dir, "Wk_bias.bin");
    p.attn.Wv.weights = load_weight(normalized_dir, "Wv_weight.bin");
    p.attn.Wv.bias = load_weight(normalized_dir, "Wv_bias.bin");
    p.attn.Wo.weights = load_weight(normalized_dir, "Wo_weight.bin");
    p.attn.Wo.bias = load_weight(normalized_dir, "Wo_bias.bin");
    
    // Load feedforward weights
    p.W1.weights = load_weight(normalized_dir, "W1_weight.bin");
    p.W1.bias = load_weight(normalized_dir, "W1_bias.bin");
    p.W2.weights = load_weight(normalized_dir, "W2_weight.bin");
    p.W2.bias = load_weight(normalized_dir, "W2_bias.bin");
    
    // Load LM head (bias only, weights are tied)
    p.lm_head.bias = load_weight(normalized_dir, "lm_head_bias.bin");
    p.lm_head.weights = mat_copy(&p.token_embed.weight_matrix);
    
    logger("✓ All weights loaded successfully!\n", INFO);
    return p;
}