#!/bin/sh
set -e

echo "=== Building for FreeDOS (32-bit x86) - Deployment ==="

mkdir -p build
cd build

echo "Assembling..."
# Define COFF for conditional assembly
# nasm -f coff -dCOFF ../src/matmul.asm -o matmul.o

echo "Compiling with DJGPP for DOS..."
SOURCES="../src/activations.c ../src/exceptions.c ../src/loader.c ../src/layers.c ../src/logger.c ../src/retrolm.c ../src/matrix.c ../src/matrix_ops.c ../src/memory.c ../src/transformer.c ../src/utils.c ../src/sampling.c ../src/chat.c"
CFLAGS="-Wall -Wextra -O3 -march=i686 -ffast-math -funroll-loops"

i586-pc-msdosdjgpp-gcc $CFLAGS $SOURCES -o retrolm.exe -lm

echo "Copying weight files..."
cp ../torch_code/weights/*.bin . 2>/dev/null || echo "No .bin files found in torch_code/weights"

cd ..
echo "✅ Build successful! Executable: build/retrolm.exe"