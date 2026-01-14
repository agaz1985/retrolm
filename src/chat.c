#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat.h"
#include "layers.h"
#include "matrix.h"
#include "transformer.h"
#include "sampling.h"

char* generate_interactive(struct TransformerParameters *model, 
                          const char *prompt,
                          unsigned int max_tokens,
                          unsigned int vocab_size,
                          float temperature) {
    if (!model) {
        fprintf(stderr, "Error: Model is NULL\n");
        return NULL;
    }
    
    // Tokenize prompt (simple char-level tokenization)
    unsigned int prompt_len = prompt ? strlen(prompt) : 0;
    unsigned int total_capacity = prompt_len + max_tokens + 1;
    
    // Allocate token array
    unsigned int *tokens = malloc(total_capacity * sizeof(unsigned int));
    if (!tokens) {
        fprintf(stderr, "Error: Failed to allocate token buffer\n");
        return NULL;
    }
    
    // Convert prompt to tokens
    unsigned int pos = 0;
    if (prompt) {
        for (unsigned int i = 0; i < prompt_len; i++) {
            tokens[pos++] = (unsigned int)(unsigned char)prompt[i];
        }
    }
    
    // Initialize attention cache
    const unsigned int embed_dim = model->attn.Wq.weights.r;
    struct AttentionCache cache = attention_cache_new(embed_dim);
    
    // PREFILL PHASE: Process entire prompt at once
    if (prompt_len > 0) {
        struct Matrix2D_UInt input = mat_uint_new(1, prompt_len);
        if (!input.data) {
            fprintf(stderr, "Error: Failed to allocate input matrix\n");
            free(tokens);
            attention_cache_free(&cache);
            return NULL;
        }
        
        for (unsigned int i = 0; i < prompt_len; i++) {
            input.data[i] = tokens[i];
        }
        
        // Forward pass through model - cache accumulates all K,V from prompt
        struct Matrix2D logits = transformer_forward(&input, model, &cache, 0);
        
        // We don't need the logits from prefill, just warming up the cache
        mat_uint_free(&input);
        mat_free(&logits);
        
        // Print prompt
        if (prompt) {
            printf("%s", prompt);
            fflush(stdout);
        }
    }
    
    // GENERATION PHASE: Generate tokens one at a time using cache
    for (unsigned int t = 0; t < max_tokens; t++) {
        // Prepare single token input
        struct Matrix2D_UInt input = mat_uint_new(1, 1);
        if (!input.data) {
            fprintf(stderr, "Error: Failed to allocate input matrix\n");
            break;
        }
        
        // Use last generated token (or last prompt token for first generation)
        input.data[0] = tokens[pos - 1];
        
        // Forward pass with cache - only processes 1 new token!
        struct Matrix2D logits = transformer_forward(&input, model, &cache, pos - 1);
        
        if (!logits.data) {
            fprintf(stderr, "Error: Forward pass failed\n");
            mat_uint_free(&input);
            break;
        }
        
        // Get logits for the single token we just processed
        float *last_logits = logits.data;
        
        // Sample next token
        unsigned int next_token = sample_from_logits(last_logits, vocab_size, temperature);
        
        // Clean up
        mat_uint_free(&input);
        mat_free(&logits);
        
        // Stop on newline or if token is out of printable range
        if (next_token == '\n' || next_token >= 127) {
            break;
        }
        
        // Add to sequence
        tokens[pos++] = next_token;
        
        // Print token in real-time
        if (next_token >= 32 && next_token <= 126) {
            printf("%c", (char)next_token);
            fflush(stdout);
        }
    }
    
    // Clean up cache
    attention_cache_free(&cache);
    
    // Convert tokens back to string
    char *result = malloc(pos + 1);
    if (!result) {
        fprintf(stderr, "Error: Failed to allocate result string\n");
        free(tokens);
        return NULL;
    }
    
    for (unsigned int i = 0; i < pos; i++) {
        if (tokens[i] >= 32 && tokens[i] <= 126) {
            result[i] = (char)tokens[i];
        } else {
            result[i] = ' ';  // Replace non-printable with space
        }
    }
    result[pos] = '\0';
    
    free(tokens);
    return result;
}

int update_history(char *history, const char *new_text) {
    if (!history || !new_text) {
        return -1;
    }
    
    size_t history_len = strlen(history);
    size_t new_len = strlen(new_text);
    size_t total_len = history_len + new_len + 1; // +1 for space
    
    // Allocate temporary buffer for concatenation
    char *temp = malloc(total_len + 1);
    if (!temp) {
        fprintf(stderr, "Error: Failed to allocate temporary buffer\n");
        return -1;
    }
    
    temp[0] = '\0';
    
    if (history_len > 0) {
        strcpy(temp, history);
        strcat(temp, " ");
    }
    strcat(temp, new_text);
    
    size_t temp_len = strlen(temp);
    
    // Keep only the last CONTEXT_WINDOW_SIZE characters
    // Note: Caller must ensure history buffer has at least CONTEXT_WINDOW_SIZE + 1 bytes
    if (temp_len > CONTEXT_WINDOW_SIZE) {
        size_t offset = temp_len - CONTEXT_WINDOW_SIZE;
        memcpy(history, temp + offset, CONTEXT_WINDOW_SIZE);
        history[CONTEXT_WINDOW_SIZE] = '\0';
    } else {
        memcpy(history, temp, temp_len + 1); // +1 to include null terminator
    }
    
    free(temp);
    return 0;
}
