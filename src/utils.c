/**
 * @file utils.c
 * @brief Implementation of utility functions
 */

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <limits.h>
    #ifndef __DJGPP__
        #include <libgen.h>
    #endif
#endif

/**
 * @brief Print RetroLM ASCII art banner to stdout
 * 
 * Displays retro-styled banner with DOS-compatible ASCII characters.
 * Compatible with FreeDOS and vintage hardware (80x25 text mode with margins).
 */
void print_retrolm(void) {
    printf("\n");
    printf("  ====================================================================\n");
    printf("  |                                                                  |\n");
    printf("  |       ##### ##### ##### ##### #####  #    #   #                 |\n");
    printf("  |       #   # #       #   #   # #   #  #    ## ##                 |\n");
    printf("  |       ##### ###     #   ##### #   #  #    # # #                 |\n");
    printf("  |       #  #  #       #   #  #  #   #  #    #   #                 |\n");
    printf("  |       #   # #####   #   #   # #####  #### #   #                 |\n");
    printf("  |                                                                  |\n");
    printf("  ====================================================================\n");
    printf("\n");
    printf("             >> RETRO VIBES LOADED - ENTER THE MATRIX <<\n");
    printf("                        [##########] 100%%\n");
    printf("\n");
}

/**
 * @brief Get the directory where the executable is located
 * 
 * Cross-platform implementation using:
 * - Linux/Unix: readlink on /proc/self/exe
 * - Windows: GetModuleFileName
 */
int get_executable_dir(char *buffer, size_t size) {
    if (!buffer || size == 0) {
        return -1;
    }
    
#ifdef _WIN32
    // Windows implementation
    char full_path[MAX_PATH];
    DWORD len = GetModuleFileNameA(NULL, full_path, MAX_PATH);
    
    if (len == 0 || len == MAX_PATH) {
        return -1;
    }
    
    // Find the last backslash to get directory
    char *last_backslash = strrchr(full_path, '\\');
    if (!last_backslash) {
        return -1;
    }
    
    // Copy directory path (excluding the backslash)
    size_t dir_len = last_backslash - full_path;
    if (dir_len >= size) {
        return -1;
    }
    
    strncpy(buffer, full_path, dir_len);
    buffer[dir_len] = '\0';
    
#else
    // Linux/Unix implementation
    char full_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", full_path, sizeof(full_path) - 1);
    
    if (len == -1) {
        return -1;
    }
    
    full_path[len] = '\0';
    
    // Get directory name - DOS-compatible method (no libgen.h needed)
    char *last_slash = strrchr(full_path, '/');
    if (!last_slash) {
        return -1;
    }
    
    // Copy directory path (excluding the slash)
    size_t dir_len = last_slash - full_path;
    if (dir_len >= size) {
        return -1;
    }
    
    strncpy(buffer, full_path, dir_len);
    buffer[dir_len] = '\0';
#endif
    
    return 0;
}

