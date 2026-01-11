# Makefile for C ML Demo with Autograd

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iinclude
LDFLAGS = -lm

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = $(BIN_DIR)/obj

# Library source files
LIB_SRCS = matrix.c nn.c optimizer.c mnist.c autograd.c weights.c mlp_model.c
LIB_OBJS = $(LIB_SRCS:%.c=$(OBJ_DIR)/%.o)

# Application source files
APP_SRCS = train.c predict.c
APP_OBJS = $(APP_SRCS:%.c=$(OBJ_DIR)/%.o)

# Target executables
TRAIN_TARGET = $(BIN_DIR)/train
PREDICT_TARGET = $(BIN_DIR)/predict

# Default target
all: $(TRAIN_TARGET) $(PREDICT_TARGET)

# Help target
help:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║              C ML Demo - Autograd MNIST Trainer               ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  Build Targets:"
	@echo "    make              Build all executables (train + predict)"
	@echo "    make clean        Remove all build artifacts"
	@echo "    make rebuild      Clean and rebuild everything"
	@echo ""
	@echo "  Run Targets:"
	@echo "    make train        Build and run training on MNIST"
	@echo "    make predict      Run prediction on image 0"
	@echo "    make predict IMG=42   Run prediction on specific image"
	@echo ""
	@echo "  Utility Targets:"
	@echo "    make install      Copy binaries to project root"
	@echo "    make compile_commands  Generate compile_commands.json"
	@echo "    make help         Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make train              # Train the model"
	@echo "  make predict IMG=123    # Predict image 123"
	@echo ""

# Build train executable
$(TRAIN_TARGET): $(LIB_OBJS) $(OBJ_DIR)/train.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build predict executable
$(PREDICT_TARGET): $(LIB_OBJS) $(OBJ_DIR)/predict.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create directories
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build files
clean:
	rm -rf $(BIN_DIR)

# Rebuild everything
rebuild: clean all

# Run training
train: $(TRAIN_TARGET)
	$(TRAIN_TARGET) data/train-images.idx3-ubyte data/train-labels.idx1-ubyte data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte

# Run prediction on a single image
# Usage: make predict [IMG=42]
predict: $(PREDICT_TARGET)
	$(PREDICT_TARGET) data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte $(if $(IMG),$(IMG),0)

# Install binaries (copy to project root for convenience)
install:
	cp $(BIN_DIR)/train ./train
	cp $(BIN_DIR)/predict ./predict

# Generate compile_commands.json for clangd/IDE support
compile_commands:
	@echo "Generating compile_commands.json using bear..."
	@if command -v bear > /dev/null 2>&1; then \
		$(MAKE) clean && bear -- $(MAKE) all; \
		echo "✓ Generated compile_commands.json"; \
		echo "✓ Restart your editor (Zed) to pick up changes"; \
	else \
		echo "❌ bear not found. Install with:"; \
		echo "   Ubuntu/Debian: sudo apt install bear"; \
		echo "   Arch Linux:    sudo pacman -S bear"; \
		echo "   macOS:         brew install bear"; \
		exit 1; \
	fi

.PHONY: all clean rebuild train predict install compile_commands help
