#ifndef _RLM_EXCEPTIONS_H
#define _RLM_EXCEPTIONS_H

enum ErrorType {
	InvalidInput,
	IndexError,
	MemoryError,
	FileError,
};

void throw(char msg[], enum ErrorType err);

#endif // _RLM_EXCEPTIONS_H