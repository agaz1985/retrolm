#ifndef _RLM_EXCEPTIONS_H
#define _RLM_EXCEPTIONS_H

enum ErrorType {
	InvalidInput,
	IndexError,
	MemoryError,
	FileError,
	ValueError,
};

void throw(char msg[], enum ErrorType err);

#endif // _RLM_EXCEPTIONS_H