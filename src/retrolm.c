#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "transformer.h"
#include "matrix.h"
#include "utils.h"
#include "loader.h"

#define MAX_INPUT 256
#define SEQ_LEN 64
#define VOCAB_SIZE 512
#define MAX_RESPONSE 512

// Modified to return generated text
char* generate_interactive(struct TransformerParameters *model, 
                          const char *history,
                          unsigned int max_tokens) {    
    unsigned int history_len = history ? strlen(history) : 0;
    
    unsigned int buffer_size = history_len + max_tokens + 1;
    unsigned int *all_tokens = malloc(buffer_size * sizeof(unsigned int));
    
    // Copy history tokens
    unsigned int pos = 0;
    if (history) {
        for (unsigned int i = 0; i < history_len; i++) {
            all_tokens[pos++] = (unsigned int)(unsigned char)history[i];
        }
    }
    
    // Allocate buffer for generated response
    char *response = malloc(max_tokens + 1);
    unsigned int response_pos = 0;
    
    for (unsigned int t = 0; t < max_tokens; t++) {
        unsigned int current_len = pos;
        unsigned int input_len = (current_len > SEQ_LEN) ? SEQ_LEN : current_len;
        unsigned int start_pos = current_len - input_len;
        
        struct Matrix2D_UInt input = mat_uint_new(1, input_len);
        for (unsigned int i = 0; i < input_len; i++) {
            input.data[i] = all_tokens[start_pos + i];
        }
        
        struct Matrix2D logits = transformer_forward(&input, model);
        float *last_logits = logits.data + (input_len - 1) * VOCAB_SIZE;
        
        // Softmax
        float max_logit = last_logits[0];
        for (unsigned int i = 1; i < VOCAB_SIZE; i++) {
            if (last_logits[i] > max_logit) max_logit = last_logits[i];
        }
        
        float sum = 0.0f;
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            last_logits[i] = expf(last_logits[i] - max_logit);
            sum += last_logits[i];
        }
        
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            last_logits[i] /= sum;
        }
        
        // Temperature sampling with temperature = 0.8
        float temperature = 0.8f;
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            last_logits[i] = powf(last_logits[i], 1.0f / temperature);
        }
        
        // Re-normalize after temperature
        sum = 0.0f;
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            sum += last_logits[i];
        }
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            last_logits[i] /= sum;
        }
        
        // Sample from distribution
        float rand_val = (float)rand() / (float)RAND_MAX;
        float cumsum = 0.0f;
        unsigned int next_token = 0;
        
        for (unsigned int i = 0; i < VOCAB_SIZE; i++) {
            cumsum += last_logits[i];
            if (rand_val < cumsum) {
                next_token = i;
                break;
            }
        }
        
        all_tokens[pos++] = next_token;
        
        // Stop on newline
        if (next_token == '\n') {
            response[response_pos] = '\0';
            mat_uint_free(&input);
            mat_free(&logits);
            break;
        }
        
        // Save printable characters to response
        if (next_token >= 32 && next_token < 127) {
            printf("%c", (char)next_token);
            response[response_pos++] = (char)next_token;
        }
        fflush(stdout);
        
        mat_uint_free(&input);
        mat_free(&logits);
    }
    
    response[response_pos] = '\0';
    free(all_tokens);
    
    return response;
}

// Helper to manage sliding window history
void update_history(char *history, const char *new_text) {
    size_t history_len = strlen(history);
    size_t new_len = strlen(new_text);
    size_t total_len = history_len + new_len + 1; // +1 for space
    
    char *temp = malloc(total_len + 1);
    temp[0] = '\0';
    
    if (history_len > 0) {
        strcpy(temp, history);
        strcat(temp, " ");
    }
    strcat(temp, new_text);
    
    size_t temp_len = strlen(temp);
    
    // Keep only the last SEQ_LEN characters
    if (temp_len > SEQ_LEN) {
        size_t offset = temp_len - SEQ_LEN;
        strcpy(history, temp + offset);
    } else {
        strcpy(history, temp);
    }
    
    free(temp);
}

int main() {
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);
    
    // Seed random number generator for sampling
    srand(time(NULL));
    
    print_retrolm();
    struct TransformerParameters model = load_model_weights("./torch_code/weights");
    
    printf("\n============================================================\n");
    printf("RetroLM Interactive Chat (Context: %d chars)\n", SEQ_LEN);
    printf("============================================================\n");
    printf("Type 'quit' or 'exit' to end the conversation\n");
    printf("============================================================\n\n");
    
    char history[SEQ_LEN * 2 + 1]; // Extra buffer for safety
    history[0] = '\0';
    
    char input[MAX_INPUT];

    while (1) {
        printf("You: ");
        fflush(stdout);
        
        if (!fgets(input, MAX_INPUT, stdin)) {
            break;
        }
        
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') {
            input[len-1] = '\0';
            len--;
        }
        
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("\nGoodbye!\n");
            break;
        }
        
        if (len == 0) {
            continue;
        }

        // Add user input to history
        update_history(history, input);
        
        printf("Bot: ");
        fflush(stdout);
        
        // Generate response and get the text back
        char *response = generate_interactive(&model, history, 100);
        
        printf("\n");
        
        // Add bot response to history
        if (strlen(response) > 0) {
            update_history(history, response);
        }
        
        free(response);
    }
    
    transformer_free(&model);
    return 0;
}