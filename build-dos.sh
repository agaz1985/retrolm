#!/bin/sh
set -e

echo "=== Building for FreeDOS (32-bit x86) - Deployment ==="

mkdir -p build
cd build

echo "Assembling..."
# Define COFF for conditional assembly
# nasm -f coff -dCOFF ../src/matmul.asm -o matmul.o

echo "Compiling with DJGPP for DOS..."
SOURCES="../src/activations.c ../src/exceptions.c ../src/loader.c ../src/layers.c ../src/logger.c ../src/retrolm.c ../src/matrix.c ../src/matrix_ops.c ../src/memory.c ../src/transformer.c ../src/utils.c"
CFLAGS="-Wall -Wextra -O2"

i586-pc-msdosdjgpp-gcc $CFLAGS $SOURCES -o retrolm.exe -lm

cd ..
echo "✅ Build successful! Executable: build/retrolm.exe"