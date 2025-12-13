#include <stdlib.h>

#include "exceptions.h"
#include "logger.h"

void throw(char msg[], enum ErrorType err) {
	logger(msg, ERROR);
	exit(err);
}