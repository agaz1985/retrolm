/**
 * @file exceptions.c
 * @brief Implementation of error handling system
 */

#include <stdlib.h>

#include "exceptions.h"
#include "logger.h"

/**
 * @brief Throw an error and exit the program
 * 
 * Logs the error message to stderr via the logger, then terminates
 * the program with the error code. The exit code corresponds to the
 * enum value of the ErrorType.
 */
void throw(char msg[], enum ErrorType err) {
	logger(msg, ERROR);
	exit(err);
}