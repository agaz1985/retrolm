/**
 * @file sampling.c
 * @brief Implementation of text sampling functions
 */

#include <stdlib.h>
#include <math.h>
#include "sampling.h"

unsigned int sample_from_logits(float *logits, unsigned int vocab_size, float temperature) {
    // Apply temperature scaling to logits first
    for (unsigned int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Apply softmax with numerical stability (subtract max)
    float max_logit = logits[0];
    for (unsigned int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    float sum = 0.0f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    // Normalize to probabilities
    for (unsigned int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }
    
    // Sample from the distribution using cumulative probability
    float rand_val = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    
    for (unsigned int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (rand_val < cumsum) {
            return i;
        }
    }
    
    // Fallback to last token (should rarely happen due to numerical precision)
    return vocab_size - 1;
}
