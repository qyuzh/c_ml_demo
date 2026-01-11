#ifndef NN_H
#define NN_H

#include "autograd.h"
#include "matrix.h"

// Linear layer structure using Tensors for autograd
typedef struct {
    Tensor* weights;      // Weight tensor (requires_grad=1)
    Tensor* bias;         // Bias tensor (requires_grad=1)
    size_t input_size;
    size_t output_size;
} Linear;

// Create and destroy layers
Linear* linear_create(size_t input_size, size_t output_size);
void linear_free(Linear* layer);

// Forward pass - returns output tensor with computational graph
Tensor* linear_forward(Linear* layer, Tensor* input);

// Zero gradients
void linear_zero_grad(Linear* layer);

// Weight initialization
void linear_init_xavier(Linear* layer);
void linear_init_he(Linear* layer);

// Helper: Create tensor from matrix data
Tensor* tensor_from_matrix(Matrix* m, int requires_grad);

// Helper: Extract matrix data from tensor
Matrix* matrix_from_tensor(Tensor* t);

// Helper: Manually compute bias gradients after backward pass
// Call this after tensor_backward() to accumulate bias gradients
void linear_bias_backward(Matrix* output_grad, Linear* layer);

#endif // NN_H