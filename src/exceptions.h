#ifndef _RLM_EXCEPTIONS_H
#define _RLM_EXCEPTIONS_H

enum ErrorType {
	InvalidInput,
	IndexError
};

void throw(char msg[], enum ErrorType err);

#endif // _RLM_EXCEPTIONS_H