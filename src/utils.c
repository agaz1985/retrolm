/**
 * @file utils.c
 * @brief Implementation of utility functions
 */

#include <stdio.h>

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

