/**
 * @file chat.c
 * @brief Implementation of interactive chat functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat.h"
#include "matrix.h"
#include "transformer.h"
#include "sampling.h"

char* generate_interactive(struct TransformerParameters *model, 
                          const char *history,
                          unsigned int max_tokens,
                          unsigned int vocab_size) {
    if (!model) {
        return NULL;
    }
    
    unsigned int history_len = history ? strlen(history) : 0;
    // Buffer size: history characters (1 char = 1 token) + new tokens + null terminator
    unsigned int buffer_size = history_len + max_tokens + 1;
    
    // Allocate token buffer
    unsigned int *all_tokens = malloc(buffer_size * sizeof(unsigned int));
    if (!all_tokens) {
        fprintf(stderr, "Error: Failed to allocate token buffer\n");
        return NULL;
    }
    
    // Copy history tokens
    unsigned int pos = 0;
    if (history) {
        for (unsigned int i = 0; i < history_len; i++) {
            all_tokens[pos++] = (unsigned int)(unsigned char)history[i];
        }
    }
    
    // Allocate response buffer (extra byte for null terminator)
    char *response = malloc(max_tokens + 2);
    if (!response) {
        fprintf(stderr, "Error: Failed to allocate response buffer\n");
        free(all_tokens);
        return NULL;
    }
    
    unsigned int response_pos = 0;
    float temperature = 0.8f;
    
    // Generation loop
    for (unsigned int t = 0; t < max_tokens; t++) {
        unsigned int current_len = pos;
        unsigned int input_len = (current_len > CONTEXT_WINDOW_SIZE) 
                                 ? CONTEXT_WINDOW_SIZE 
                                 : current_len;
        unsigned int start_pos = current_len - input_len;
        
        // Prepare input matrix
        struct Matrix2D_UInt input = mat_uint_new(1, input_len);
        if (!input.data) {
            fprintf(stderr, "Error: Failed to allocate input matrix\n");
            free(all_tokens);
            free(response);
            return NULL;
        }
        
        for (unsigned int i = 0; i < input_len; i++) {
            input.data[i] = all_tokens[start_pos + i];
        }
        
        // Forward pass through model
        struct Matrix2D logits = transformer_forward(&input, model);
        if (!logits.data) {
            fprintf(stderr, "Error: Model forward pass failed\n");
            mat_uint_free(&input);
            free(all_tokens);
            free(response);
            return NULL;
        }
        
        // Get logits for last position
        float *last_logits = logits.data + (input_len - 1) * vocab_size;
        
        // Sample next token using temperature sampling
        unsigned int next_token = sample_from_logits(last_logits, vocab_size, temperature);
        
        // Clean up for this iteration
        mat_uint_free(&input);
        mat_free(&logits);
        
        // Add token to history
        all_tokens[pos++] = next_token;
        
        // Stop on newline
        if (next_token == '\n') {
            break;
        }
        
        // Save printable characters to response
        if (next_token >= 32 && next_token < 127) {
            printf("%c", (char)next_token);
            fflush(stdout);
            response[response_pos++] = (char)next_token;
        }
    }
    
    response[response_pos] = '\0';
    free(all_tokens);
    
    return response;
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
