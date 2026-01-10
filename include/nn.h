#ifndef NN_H
#define NN_H

#include "matrix.h"

// Linear layer structure
typedef struct {
    Matrix* weights;
    Matrix* bias;
    Matrix* weight_grad;
    Matrix* bias_grad;
    Matrix* input;  // Cached for backward pass
} Linear;

// Create and destroy layers
Linear* linear_create(size_t input_size, size_t output_size);
void linear_free(Linear* layer);

// Forward and backward passes
Matrix* linear_forward(Linear* layer, const Matrix* input);
void linear_backward(Linear* layer, const Matrix* grad_output, Matrix** grad_input);
void linear_zero_grad(Linear* layer);

// Initialize weights
void linear_init_xavier(Linear* layer);
void linear_init_he(Linear* layer);

#endif // NN_H
