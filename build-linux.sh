#!/bin/sh
set -e

echo "=== Building for Linux (32-bit x86) - Development ==="

mkdir -p build
cd build

echo "Assembling..."
#nasm -f elf32 ../src/matmul.asm -o matmul.o

echo "Compiling C sources..."
SOURCES="../src/activations.c ../src/exceptions.c ../src/loader.c ../src/layers.c ../src/logger.c ../src/retrolm.c ../src/matrix.c ../src/matrix_ops.c ../src/memory.c ../src/transformer.c ../src/utils.c ../src/sampling.c ../src/chat.c"
CFLAGS="-m32 -Wall -Wextra -O2"

gcc $CFLAGS $SOURCES -lm -o retrolm

cd ..
echo "✅ Build successful! Binary: build/retrolm"
