#ifndef _RLM_LOGGER_H
#define _RLM_LOGGER_H

enum LogLevel {
	DEBUG,
	INFO,
	WARNING,
	NONE,
	ERROR
};

#define LOG_LEVEL DEBUG

void logger(char msg[], enum LogLevel level);

#endif // _RLM_LOGGER_H