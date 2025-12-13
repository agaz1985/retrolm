#!/bin/sh
echo "=== Entering Docker container shell ==="
echo "Use 'exit' to leave the container"
docker run --rm -it -v "$(pwd):/project" retrolm-builder bash
