#!/bin/sh

mkdir build
cd build
gcc -o retrollm ../src/exceptions.c ../src/logger.c ../src/main.c ../src/matrix.c ../src/memory.c
cd ..