.PHONY: build dev dos run shell clean help tests

IMAGE_NAME := retrolm-builder

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .
	@echo "✅ Docker image ready"

# Build for Linux (development)
dev: build
	docker run --rm -v "$$(pwd):/project" $(IMAGE_NAME) ./build-linux.sh

# Build for DOS (deployment)
dos: build
	docker run --rm -v "$$(pwd):/project" $(IMAGE_NAME) ./build-dos.sh

# Build and run Linux version
run: dev
	@echo ""
	@echo "=== Running Linux binary ==="
	docker run -it --rm -v "$$(pwd):/project" $(IMAGE_NAME) ./build/retrolm ./build/

# Compile and run tests
tests: build
	@echo ""
	@echo "=== Compiling and running unit tests ==="
	@mkdir -p build
	docker run --rm -v "$$(pwd):/project" $(IMAGE_NAME) bash -c "\
		mkdir -p build && \
		gcc -Wall -Wextra -std=c99 -o build/test_runner \
			tests/test_runner.c \
			tests/test_matrix.c \
			tests/test_matrix_ops.c \
			tests/test_activations.c \
			tests/test_memory.c \
			tests/test_utils.c \
			tests/test_sampling.c \
			tests/test_layers.c \
			tests/test_transformer.c \
			src/matrix.c \
			src/matrix_ops.c \
			src/activations.c \
			src/memory.c \
			src/exceptions.c \
			src/logger.c \
			src/utils.c \
			src/sampling.c \
			src/layers.c \
			src/transformer.c -lm && \
		./build/test_runner"

# Open interactive shell in container
shell: build
	docker run --rm -it -v "$$(pwd):/project" $(IMAGE_NAME) bash

# Clean build artifacts
clean:
	rm -rf build

# Clean Docker image
clean-docker:
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

# Show help
help:
	@echo "Available commands:"
	@echo "  make dev    - Build for Linux (fast testing)"
	@echo "  make dos    - Build for FreeDOS (deployment)"
	@echo "  make run    - Build and run Linux version"
	@echo "  make tests  - Compile and run unit tests"
	@echo "  make shell  - Open Docker shell for debugging"
	@echo "  make clean  - Remove build directory"
	@echo "  make clean-docker - Remove Docker image"