/**
 * @file sampling.h
 * @brief Text sampling functions for language model generation
 * 
 * Provides temperature-based sampling from probability distributions
 */

#ifndef _RLM_SAMPLING_H
#define _RLM_SAMPLING_H

/**
 * @brief Sample from logits using temperature sampling
 * 
 * Applies temperature scaling and softmax normalization to convert
 * raw logits into a probability distribution, then samples from it.
 * 
 * Process:
 * 1. Apply softmax to convert logits to probabilities
 * 2. Apply temperature scaling (higher temp = more random)
 * 3. Re-normalize probabilities
 * 4. Sample using cumulative distribution
 * 
 * @param logits Array of raw logit values
 * @param vocab_size Size of vocabulary (length of logits array)
 * @param temperature Sampling temperature (0.1-2.0 typical range)
 *                    Lower = more deterministic, higher = more random
 * @return Sampled token index from vocabulary
 * 
 * @note Modifies the logits array in-place for efficiency
 * @note Uses global rand() for sampling - seed with srand() before use
 */
unsigned int sample_from_logits(float *logits, unsigned int vocab_size, float temperature);

#endif // _RLM_SAMPLING_H
