/**
 * @file exceptions.h
 * @brief Error handling and exception management system
 * 
 * This module provides a simple exception handling mechanism for the RetroLM project.
 * It defines error types and a throw function that logs errors and terminates execution.
 */

#ifndef _RLM_EXCEPTIONS_H
#define _RLM_EXCEPTIONS_H

/**
 * @brief Error type enumeration
 * 
 * Defines the different types of errors that can occur in the system.
 * Each error type has a distinct numeric value that becomes the exit code.
 */
enum ErrorType {
	InvalidInput,   /**< Invalid input parameters or data */
	IndexError,     /**< Array or matrix index out of bounds */
	MemoryError,    /**< Memory allocation failure */
	FileError,      /**< File I/O operation failure */
};

/**
 * @brief Throw an error and terminate the program
 * 
 * This function logs an error message using the logger system and then
 * terminates the program with the error type as the exit code.
 * 
 * @param msg Error message to display (null-terminated string)
 * @param err Type of error being thrown
 * 
 * @note This function does not return - it terminates the program
 * @note The error message is logged at ERROR level before termination
 */
void throw(char msg[], enum ErrorType err);

#endif // _RLM_EXCEPTIONS_H