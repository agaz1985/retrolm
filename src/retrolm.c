#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "transformer.h"
#include "matrix.h"
#include "utils.h"
#include "loader.h"

#define MAX_INPUT 256
#define SEQ_LEN 8

void generate_interactive(struct TransformerParameters *model, 
                          const char *prompt,
                          const char *history,
                          unsigned int max_tokens) {
    const unsigned int VOCAB_SIZE = 128;
    
    unsigned int history_len = history ? strlen(history) : 0;
    unsigned int prompt_len = strlen(prompt);
    unsigned int total_input_len = history_len + prompt_len;
    
    unsigned int buffer_size = total_input_len + max_tokens + 1;
    unsigned int *all_tokens = malloc(buffer_size * sizeof(unsigned int));
    
    unsigned int pos = 0;
    if (history) {
        for (unsigned int i = 0; i < history_len; i++) {
            all_tokens[pos++] = (unsigned int)(unsigned char)history[i];
        }
    }
    
    for (unsigned int i = 0; i < prompt_len; i++) {
        all_tokens[pos++] = (unsigned int)(unsigned char)prompt[i];
    }
    
    printf("%s", prompt);
    fflush(stdout);
    
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
        
        unsigned int next_token = 0;
        float max_prob = last_logits[0];
        for (unsigned int i = 1; i < VOCAB_SIZE; i++) {
            if (last_logits[i] > max_prob) {
                max_prob = last_logits[i];
                next_token = i;
            }
        }
        
        all_tokens[pos++] = next_token;
        
        if (next_token == '\n') {
            printf("\n");
            mat_uint_free(&input);
            mat_free(&logits);
            break;
        }
        
        if (next_token >= 32 && next_token < 127) {
            printf("%c", (char)next_token);
        }
        fflush(stdout);
        
        mat_uint_free(&input);
        mat_free(&logits);
    }
    
    free(all_tokens);
}

int main() {
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);
    
    print_retrolm();
    struct TransformerParameters model = load_model_weights("./torch_code/weights");
    
    printf("\n============================================================\n");
    printf("RetroLM Interactive Chat (Context: %d chars)\n", SEQ_LEN);
    printf("============================================================\n");
    printf("Type 'quit' or 'exit' to end the conversation\n");
    printf("============================================================\n\n");
    
    char history[SEQ_LEN + 1];
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

        char temp_buffer[SEQ_LEN + MAX_INPUT + 2];
        temp_buffer[0] = '\0';
        
        if (strlen(history) > 0) {
            strcpy(temp_buffer, history);
            strcat(temp_buffer, " ");
        }
        strcat(temp_buffer, input);
        
        size_t temp_len = strlen(temp_buffer);
        
        if (temp_len > SEQ_LEN) {
            size_t offset = temp_len - SEQ_LEN;
            strcpy(history, temp_buffer + offset);
        } else {
            strcpy(history, temp_buffer);
        }
        
        printf("Bot: ");
        fflush(stdout);
        
        generate_interactive(&model, "", history, 100);
        
        printf("\n");
    }
    
    transformer_free(&model);
    return 0;
}