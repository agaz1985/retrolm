#include <stdlib.h>
#include <math.h>
#include "sampling.h"

unsigned int sample_from_logits(float *logits, unsigned int vocab_size, float temperature) {
    // Validate temperature (must be positive)
    if (temperature <= 0.0f) {
        temperature = 1.0f;  // Default to 1.0 for invalid values
    }
    
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (unsigned int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Allocate temporary array for probabilities
    float *probs = malloc(vocab_size * sizeof(float));
    if (!probs) {
        // Fallback: return first token if allocation fails
        return 0;
    }
    
    // Apply temperature scaling and softmax
    float sum = 0.0f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    // Normalize to probabilities
    for (unsigned int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Sample from the distribution using cumulative probability
    float rand_val = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    
    unsigned int result = vocab_size - 1;  // Default fallback
    for (unsigned int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            result = i;
            break;
        }
    }
    
    free(probs);
    return result;
}
