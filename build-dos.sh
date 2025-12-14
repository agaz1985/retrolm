#!/bin/sh
set -e

echo "=== Building for FreeDOS (32-bit x86) - Deployment ==="

mkdir -p build
cd build

echo "Assembling..."
# Define COFF for conditional assembly
# nasm -f coff -dCOFF ../src/matmul.asm -o matmul.o

echo "Compiling with DJGPP for DOS..."
SOURCES="../src/exceptions.c ../src/logger.c ../src/main.c ../src/matrix.c ../src/matrix_ops.c ../src/memory.c"
CFLAGS="-Wall -Wextra -O2"

i586-pc-msdosdjgpp-gcc $CFLAGS $SOURCES matmul.o -o retrollm.exe

cd ..
echo "✅ Build successful! Executable: build/retrollm.exe"