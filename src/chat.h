/**
 * @file chat.h
 * @brief Interactive chat functions for RetroLM
 * 
 * Provides high-level functions for interactive text generation
 * and conversation history management
 */

#ifndef _RLM_CHAT_H
#define _RLM_CHAT_H

#include "transformer.h"

/** Maximum tokens to generate in a single response */
#define MAX_RESPONSE_TOKENS 512

/** Context window size for sliding history (kept small since no KV caching) */
#define CONTEXT_WINDOW_SIZE 16

/**
 * @brief Generate text interactively using the transformer model
 * 
 * Performs autoregressive text generation with sliding window context
 * and temperature-based sampling. Streams output to stdout in real-time.
 * 
 * Features:
 * - Sliding window context (last CONTEXT_WINDOW_SIZE tokens)
 * - Temperature-based sampling (temp=0.8)
 * - Early stopping on newline character
 * - Real-time output streaming
 * - Filters for printable ASCII characters only
 * 
 * @param model Pointer to loaded transformer model
 * @param history Context string (can be NULL)
 * @param max_tokens Maximum number of tokens to generate
 * @param vocab_size Size of the vocabulary
 * @return Dynamically allocated string with generated text (caller must free)
 *         Returns NULL on memory allocation failure
 * 
 * @note Prints generated tokens to stdout in real-time
 * @note Only printable ASCII characters (32-126) are included in output
 */
char* generate_interactive(struct TransformerParameters *model, 
                          const char *history,
                          unsigned int max_tokens,
                          unsigned int vocab_size);

/**
 * @brief Update conversation history with sliding window
 * 
 * Appends new text to history and truncates to CONTEXT_WINDOW_SIZE characters.
 * Maintains a fixed-size context window for the model.
 * 
 * @param history Buffer to update (must be at least CONTEXT_WINDOW_SIZE*2+1 bytes)
 * @param new_text Text to append to history
 * @return 0 on success, -1 on memory allocation failure
 * 
 * @note Keeps only the most recent CONTEXT_WINDOW_SIZE characters
 * @note Automatically manages memory and truncation
 */
int update_history(char *history, const char *new_text);

#endif // _RLM_CHAT_H
