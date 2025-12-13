#include <stdio.h>
#include <time.h>

#include "logger.h"

void logger(char msg[], enum LogLevel level) {
	const char* prefix;

	if (level == DEBUG) {
		prefix = "DEBUG";
	} else if (level == INFO) {
		prefix = "INFO";
	} else if (level == WARNING) {
		prefix = "WARNING";
	} else {
		prefix = "ERROR";
	}

	if (level >= LOG_LEVEL) {
		time_t now = time(NULL);
		struct tm *t = localtime(&now);
		char buffer[100];
		strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", t);

		if (level == ERROR) {
			fprintf(stderr, "%s | %s: %s", buffer, prefix, msg);
		} else {
			printf("%s | %s: %s", buffer, prefix, msg);
		}
	}
}