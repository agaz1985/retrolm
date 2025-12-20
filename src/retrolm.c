/**
 * @file retrolm.c
 * @brief Main entry point for RetroLM interactive chat application
 * 
 * This program implements an interactive chatbot using a transformer language model.
 * It supports:
 * - Loading pre-trained model weights
 * - Interactive text generation with temperature sampling
 * - Sliding window context management
 * - Real-time streaming output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "transformer.h"
#include "utils.h"
#include "loader.h"
#include "chat.h"

#define MAX_INPUT 256      /**< Maximum input length from user */
#define VOCAB_SIZE 512     /**< Vocabulary size (ASCII-based) */

/**
 * @brief Main entry point for RetroLM interactive chat
 * 
 * Program flow:
 * 1. Initialize random seed for sampling
 * 2. Print banner
 * 3. Load pre-trained model weights
 * 4. Enter interactive loop:
 *    - Read user input
 *    - Update context history
 *    - Generate response
 *    - Display and update history with response
 * 5. Clean up and exit
 * 
 * @return 0 on success, 1 on failure
 * 
 * @note Expects model weights in ./torch_code/weights/ directory
 * @note Type 'quit' or 'exit' to end the conversation
 */
int main(void) {
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);
    
    // Seed random number generator for sampling
    srand((unsigned int)time(NULL));
    
    // Print welcome banner
    print_retrolm();
    
    // Load model weights
    struct TransformerParameters model = load_model_weights("./torch_code/weights");
    
    // Note: load_model_weights will call exit() on failure, so no need to check here
    // In a future improvement, it should return an error code instead
    
    printf("\n============================================================\n");
    printf("RetroLM Interactive Chat (Context: %d chars)\n", CONTEXT_WINDOW_SIZE);
    printf("============================================================\n");
    printf("Type 'quit' or 'exit' to end the conversation\n");
    printf("============================================================\n\n");
    
    // Initialize conversation history buffer
    // Size: CONTEXT_WINDOW_SIZE for content + extra space for intermediate operations + null terminator
    // update_history() requires at least CONTEXT_WINDOW_SIZE + 1 bytes
    char history[CONTEXT_WINDOW_SIZE * 2 + 1];
    history[0] = '\0';
    
    char input[MAX_INPUT];
    int exit_code = 0;

    while (1) {
        printf("You: ");
        fflush(stdout);
        
        // Read user input
        if (!fgets(input, MAX_INPUT, stdin)) {
            break;
        }
        
        // Remove trailing newline
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') {
            input[len-1] = '\0';
            len--;
        }
        
        // Check for exit commands
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("\nGoodbye!\n");
            break;
        }
        
        // Skip empty input
        if (len == 0) {
            continue;
        }

        // Add user input to history
        if (update_history(history, input) != 0) {
            fprintf(stderr, "Warning: Failed to update history with user input\n");
        }
        
        printf("Bot: ");
        fflush(stdout);
        
        // Generate response (100 tokens max)
        char *response = generate_interactive(&model, history, 100, VOCAB_SIZE);
        
        if (!response) {
            fprintf(stderr, "\nError: Failed to generate response\n");
            exit_code = 1;
            break;
        }
        
        printf("\n");
        
        // Add bot response to history
        if (strlen(response) > 0) {
            if (update_history(history, response) != 0) {
                fprintf(stderr, "Warning: Failed to update history with bot response\n");
            }
        }
        
        free(response);
    }
    
    // Clean up model resources
    transformer_free(&model);
    
    return exit_code;
}