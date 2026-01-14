#ifndef _RLM_CHAT_H
#define _RLM_CHAT_H

#include "transformer.h"

#define MAX_RESPONSE_TOKENS 512

#define CONTEXT_WINDOW_SIZE 16

char* generate_interactive(struct TransformerParameters *model, 
                          const char *prompt,
                          unsigned int max_tokens,
                          unsigned int vocab_size,
                          float temperature);

int update_history(char *history, const char *new_text);

#endif // _RLM_CHAT_H
