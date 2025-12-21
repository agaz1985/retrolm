.PHONY: build dev dos run shell clean help

IMAGE_NAME := retrolm-builder

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .
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
	@echo "  make shell  - Open Docker shell for debugging"
	@echo "  make clean  - Remove build directory"
	@echo "  make clean-docker - Remove Docker image"