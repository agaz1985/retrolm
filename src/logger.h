/**
 * @file logger.h
 * @brief Simple logging system with multiple severity levels
 * 
 * Provides a basic logging facility that outputs timestamped messages to stdout/stderr
 * based on configured log levels. Messages below the configured LOG_LEVEL are suppressed.
 */

#ifndef _RLM_LOGGER_H
#define _RLM_LOGGER_H

/**
 * @brief Log level enumeration
 * 
 * Defines severity levels for log messages. Only messages with level >= LOG_LEVEL
 * will be printed.
 */
enum LogLevel {
	DEBUG,    /**< Detailed diagnostic information */
	INFO,     /**< General informational messages */
	WARNING,  /**< Warning messages for potential issues */
	NONE,     /**< Placeholder level (no logging) */
	ERROR     /**< Error messages (printed to stderr) */
};

/**
 * @brief Current log level threshold
 * 
 * Change this macro to control verbosity. Messages with level < LOG_LEVEL are suppressed.
 * DEBUG: Show all messages
 * INFO: Show INFO, WARNING, ERROR
 * WARNING: Show WARNING, ERROR
 * ERROR: Show only ERROR
 */
#define LOG_LEVEL DEBUG

/**
 * @brief Log a message with specified severity level
 * 
 * Prints a timestamped message if the level meets the LOG_LEVEL threshold.
 * ERROR level messages are printed to stderr, others to stdout.
 * 
 * @param msg Message string to log (null-terminated)
 * @param level Severity level of the message
 * 
 * @note Format: "DD-MM-YYYY HH:MM:SS | LEVEL: message"
 * @note Output is flushed immediately after printing
 */
void logger(char msg[], enum LogLevel level);

#endif // _RLM_LOGGER_H