#ifndef _RLM_SAMPLING_H
#define _RLM_SAMPLING_H

unsigned int sample_from_logits(float *logits, unsigned int vocab_size, float temperature);

#endif // _RLM_SAMPLING_H
