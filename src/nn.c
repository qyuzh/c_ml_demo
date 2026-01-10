#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Create and destroy layers
Linear* linear_create(size_t input_size, size_t output_size) {
  Linear* layer = (Linear*)malloc(sizeof(Linear));

  layer->weights = matrix_create(input_size, output_size);
  layer->bias = matrix_create(1, output_size);
  layer->weight_grad = matrix_create(input_size, output_size);
  layer->bias_grad = matrix_create(1, output_size);
  layer->input = NULL;

  // Initialize gradients to zero
  matrix_zeros(layer->weight_grad);
  matrix_zeros(layer->bias_grad);

  return layer;
}

void linear_free(Linear* layer) {
  if (layer) {
    if (layer->weights) matrix_free(layer->weights);
    if (layer->bias) matrix_free(layer->bias);
    if (layer->weight_grad) matrix_free(layer->weight_grad);
    if (layer->bias_grad) matrix_free(layer->bias_grad);
    if (layer->input) matrix_free(layer->input);
    free(layer);
  }
}

// Forward pass: output = input @ weights + bias
Matrix* linear_forward(Linear* layer, const Matrix* input) {
  // Cache input for backward pass
  if (layer->input) {
    matrix_free(layer->input);
  }
  layer->input = matrix_copy(input);

  // Compute: output = input @ weights
  Matrix* output = matrix_matmul(input, layer->weights);

  // Add bias to each row
  for (size_t i = 0; i < output->rows; i++) {
    for (size_t j = 0; j < output->cols; j++) {
      output->data[i * output->cols + j] += layer->bias->data[j];
    }
  }

  return output;
}

// Backward pass
void linear_backward(Linear* layer, const Matrix* grad_output,
                     Matrix** grad_input) {
  // Compute weight gradient: input^T @ grad_output
  Matrix* input_t = matrix_transpose(layer->input);
  Matrix* weight_grad = matrix_matmul(input_t, grad_output);

  // Accumulate weight gradients
  matrix_add_inplace(layer->weight_grad, weight_grad);
  matrix_free(weight_grad);
  matrix_free(input_t);

  // Compute bias gradient: sum over batch dimension
  for (size_t j = 0; j < grad_output->cols; j++) {
    float sum = 0.0f;
    for (size_t i = 0; i < grad_output->rows; i++) {
      sum += grad_output->data[i * grad_output->cols + j];
    }
    layer->bias_grad->data[j] += sum;
  }

  // Compute input gradient: grad_output @ weights^T
  if (grad_input != NULL) {
    Matrix* weights_t = matrix_transpose(layer->weights);
    *grad_input = matrix_matmul(grad_output, weights_t);
    matrix_free(weights_t);
  }
}

void linear_zero_grad(Linear* layer) {
  matrix_zeros(layer->weight_grad);
  matrix_zeros(layer->bias_grad);
}

// Initialize weights using Xavier initialization
void linear_init_xavier(Linear* layer) {
  size_t fan_in = layer->weights->rows;
  size_t fan_out = layer->weights->cols;
  float limit = sqrtf(6.0f / (fan_in + fan_out));

  matrix_random(layer->weights, -limit, limit);
  matrix_zeros(layer->bias);
}

// Initialize weights using He initialization (good for ReLU)
void linear_init_he(Linear* layer) {
  size_t fan_in = layer->weights->rows;
  float std = sqrtf(2.0f / fan_in);

  matrix_random(layer->weights, -std, std);
  matrix_zeros(layer->bias);
}
