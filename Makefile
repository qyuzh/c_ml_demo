CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -Iinclude
LDFLAGS = -lm

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = $(BIN_DIR)/obj

# Library source files
LIB_SRCS = matrix.c nn.c optimizer.c mnist.c autograd.c weights.c
LIB_OBJS = $(LIB_SRCS:%.c=$(OBJ_DIR)/%.o)

# Application source files
APP_SRCS = train.c test.c example.c predict.c
APP_OBJS = $(APP_SRCS:%.c=$(OBJ_DIR)/%.o)

# Target executables
TRAIN_TARGET = $(BIN_DIR)/train
TEST_TARGET = $(BIN_DIR)/test
EXAMPLE_TARGET = $(BIN_DIR)/example
PREDICT_TARGET = $(BIN_DIR)/predict

# Default target
all: $(TRAIN_TARGET) $(TEST_TARGET) $(EXAMPLE_TARGET) $(PREDICT_TARGET)

# Build train executable
$(TRAIN_TARGET): $(LIB_OBJS) $(OBJ_DIR)/train.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build test executable
$(TEST_TARGET): $(LIB_OBJS) $(OBJ_DIR)/test.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build example executable
$(EXAMPLE_TARGET): $(LIB_OBJS) $(OBJ_DIR)/example.o | $(BIN_DIR)
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

# Run example
example_run: $(EXAMPLE_TARGET)
	$(EXAMPLE_TARGET)

# Run tests
test_run: $(TEST_TARGET)
	$(TEST_TARGET)

# Run training
run: $(TRAIN_TARGET)
	$(TRAIN_TARGET) data/train-images.idx3-ubyte data/train-labels.idx1-ubyte data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte

# Install binaries (copy to project root for convenience)
install:
	cp $(BIN_DIR)/train ./train
	cp $(BIN_DIR)/test ./test
	cp $(BIN_DIR)/example ./example
	cp $(BIN_DIR)/predict ./predict

.PHONY: all clean rebuild example_run test_run run install
