#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Create and destroy layers
Linear* linear_create(size_t input_size, size_t output_size) {
  Linear* layer = (Linear*)malloc(sizeof(Linear));
  
  layer->input_size = input_size;
  layer->output_size = output_size;
  
  // Create weight tensor (input_size x output_size) with gradient tracking
  Matrix* weight_data = matrix_create(input_size, output_size);
  layer->weights = tensor_create(weight_data, 1);  // requires_grad = 1
  
  // Create bias tensor (1 x output_size) with gradient tracking
  Matrix* bias_data = matrix_create(1, output_size);
  layer->bias = tensor_create(bias_data, 1);  // requires_grad = 1
  
  return layer;
}

void linear_free(Linear* layer) {
  if (layer) {
    if (layer->weights) tensor_free(layer->weights);
    if (layer->bias) tensor_free(layer->bias);
    free(layer);
  }
}

// Forward pass: output = input @ weights + bias
// Weight gradients are handled by autograd (tensor_matmul)
// Bias gradients are computed manually after backward pass
Tensor* linear_forward(Linear* layer, Tensor* input) {
  // Matrix multiplication: input @ weights
  // This automatically builds the computational graph for weight gradients
  Tensor* matmul_out = tensor_matmul(input, layer->weights);
  
  // Add bias manually (broadcast across batch dimension)
  // We don't use tensor operations here to avoid dimension mismatch in gradients
  Matrix* output_data = matrix_create(matmul_out->data->rows, matmul_out->data->cols);
  
  // Copy matmul result and add bias to each row
  for (size_t i = 0; i < output_data->rows; i++) {
    for (size_t j = 0; j < output_data->cols; j++) {
      output_data->data[i * output_data->cols + j] = 
          matmul_out->data->data[i * matmul_out->data->cols + j] + 
          layer->bias->data->data[j];
    }
  }
  
  // Create output tensor
  // The output tensor will track gradients through matmul_out (for weights)
  // Bias gradients will be computed manually
  int requires_grad = input->requires_grad || layer->weights->requires_grad;
  Tensor* output = tensor_create(output_data, requires_grad);
  
  // Set up computational graph for weight gradients only
  // The gradient will flow back through matmul_out to weights automatically
  if (requires_grad) {
    output->op = OP_ADD;
    output->parent1 = matmul_out;
    output->parent2 = NULL;  // No autograd for bias
  }
  
  return output;
}

// Manually compute bias gradients after backward pass
// Call this after tensor_backward() completes
// output_grad: the gradient at the layer output (batch_size x output_size)
// layer: the layer whose bias gradient we want to accumulate
void linear_bias_backward(Matrix* output_grad, Linear* layer) {
  if (!layer->bias->requires_grad || !output_grad) {
    return;
  }
  
  // Bias gradient = sum of output gradients across batch dimension
  // output_grad has shape (batch_size x output_size)
  // bias->grad has shape (1 x output_size)
  
  for (size_t j = 0; j < output_grad->cols; j++) {
    float sum = 0.0f;
    for (size_t i = 0; i < output_grad->rows; i++) {
      sum += output_grad->data[i * output_grad->cols + j];
    }
    layer->bias->grad->data[j] += sum;
  }
}

// Zero gradients
void linear_zero_grad(Linear* layer) {
  tensor_zero_grad(layer->weights);
  tensor_zero_grad(layer->bias);
}

// Initialize weights using Xavier initialization
void linear_init_xavier(Linear* layer) {
  size_t fan_in = layer->input_size;
  size_t fan_out = layer->output_size;
  float limit = sqrtf(6.0f / (fan_in + fan_out));
  
  matrix_random(layer->weights->data, -limit, limit);
  matrix_zeros(layer->bias->data);
}

// Initialize weights using He initialization (good for ReLU)
void linear_init_he(Linear* layer) {
  size_t fan_in = layer->input_size;
  float std = sqrtf(2.0f / fan_in);
  
  matrix_random(layer->weights->data, -std, std);
  matrix_zeros(layer->bias->data);
}

// Helper: Create tensor from matrix data (copies the data)
Tensor* tensor_from_matrix(Matrix* m, int requires_grad) {
  Matrix* data = matrix_copy(m);
  return tensor_create(data, requires_grad);
}

// Helper: Extract matrix data from tensor (copies the data)
Matrix* matrix_from_tensor(Tensor* t) {
  return matrix_copy(t->data);
}