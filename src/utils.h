/**
 * @file utils.h
 * @brief Utility functions for RetroLM
 */

#ifndef _RLM_UTILS_H
#define _RLM_UTILS_H

#include <stddef.h>

/**
 * @brief Print RetroLM ASCII art banner
 * 
 * Displays a retro-styled ASCII art banner with the RetroLM logo.
 * Used for startup messages and visual flair.
 */
void print_retrolm(void);

/**
 * @brief Get the directory where the executable is located
 * 
 * Cross-platform function that returns the directory path of the currently
 * running executable. Works on both Linux (using /proc/self/exe) and
 * Windows (using GetModuleFileName).
 * 
 * @param buffer Buffer to store the directory path
 * @param size Size of the buffer
 * @return 0 on success, -1 on failure
 * 
 * @note The buffer should be at least 256 bytes for typical paths
 * @note On DOS/Windows, uses backslashes in paths; on Linux, uses forward slashes
 */
int get_executable_dir(char *buffer, size_t size);

#endif // _RLM_UTILS_H