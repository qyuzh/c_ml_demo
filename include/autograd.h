#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "matrix.h"

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_MATMUL,
    OP_RELU,
    OP_SIGMOID,
    OP_SOFTMAX
} OpType;

typedef struct Tensor Tensor;

struct Tensor {
    Matrix* data;
    Matrix* grad;
    OpType op;
    Tensor* parent1;
    Tensor* parent2;
    int requires_grad;
};

// Tensor creation and destruction
Tensor* tensor_create(Matrix* data, int requires_grad);
Tensor* tensor_create_from_shape(size_t rows, size_t cols, int requires_grad);
void tensor_free(Tensor* t);

// Forward operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_sigmoid(Tensor* a);
Tensor* tensor_softmax(Tensor* a);

// Backward operations
void tensor_backward(Tensor* t);
void tensor_zero_grad(Tensor* t);

// Utility functions
void tensor_print(const Tensor* t);

#endif // AUTOGRAD_H
