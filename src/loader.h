/**
 * @file loader.h
 * @brief Model weight loading utilities
 * 
 * Provides functions to load pre-trained model weights from binary files.
 * The binary format stores dimensions (rows, cols) followed by float32 data.
 * This module is responsible for deserializing trained PyTorch models exported
 * to the RetroLM binary format.
 */

#ifndef _RLM_LOADER_H
#define _RLM_LOADER_H

#include "matrix.h"
#include "transformer.h"

/**
 * @brief Load a single weight matrix from a binary file
 * 
 * Binary file format:
 * - 4 bytes: unsigned int rows
 * - 4 bytes: unsigned int cols
 * - rows * cols * 4 bytes: float data (row-major order)
 * 
 * @param filepath Path to binary weight file
 * @return Loaded matrix with data from file
 * 
 * @throws FileError if file cannot be opened or read
 * @note The returned matrix must be freed with mat_free()
 */
struct Matrix2D load_weight_matrix(const char *filepath);

/**
 * @brief Load all transformer model weights from a directory
 * 
 * Loads all weight matrices for a complete transformer model from binary files
 * in the specified directory. Implements weight tying by sharing token embeddings
 * with the language model head.
 * 
 * Expected files in weights_dir:
 * - token_embed.bin: Token embedding matrix [vocab_size x embed_dim]
 * - pos_embed.bin: Positional embeddings [seq_len x embed_dim]
 * - Wq_weight.bin, Wq_bias.bin: Query projection weights and bias
 * - Wk_weight.bin, Wk_bias.bin: Key projection weights and bias
 * - Wv_weight.bin, Wv_bias.bin: Value projection weights and bias
 * - Wo_weight.bin, Wo_bias.bin: Output projection weights and bias
 * - W1_weight.bin, W1_bias.bin: First feed-forward layer
 * - W2_weight.bin, W2_bias.bin: Second feed-forward layer
 * - lm_head_bias.bin: Language model head bias
 * 
 * @param weights_dir Path to directory containing weight files
 * @return TransformerParameters with all weights loaded
 * 
 * @throws FileError if any weight file cannot be loaded
 * @note The returned parameters must be freed with transformer_free()
 * @note Logs loading progress at INFO and DEBUG levels
 */
struct TransformerParameters load_model_weights(const char *weights_dir);

#endif // _RLM_LOADER_H