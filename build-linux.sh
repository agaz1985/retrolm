#!/bin/sh
set -e

echo "=== Building for Linux (32-bit x86) - Development ==="

mkdir -p build
cd build

echo "Assembling..."
#nasm -f elf32 ../src/matmul.asm -o matmul.o

echo "Compiling C sources..."
SOURCES="../src/exceptions.c ../src/logger.c ../src/main.c ../src/matrix.c ../src/matrix_ops.c ../src/memory.c"
CFLAGS="-m32 -Wall -Wextra -O2"

gcc $CFLAGS $SOURCES matmul.o -o retrollm

cd ..
echo "✅ Build successful! Binary: build/retrollm"
