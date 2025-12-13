#ifndef _RLM_EXCEPTIONS_H
#define _RLM_EXCEPTIONS_H

enum ErrorType {
	InvalidInput
};

void throw(char msg[], enum ErrorType err);

#endif // _RLM_EXCEPTIONS_H