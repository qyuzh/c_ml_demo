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

  // ============================================================================
  // PARAMETER INITIALIZATION
  // ============================================================================
  // Create weight tensor: [input_size, output_size] with gradient tracking
  // Example: For fc1 (784 \u2192 128), weights shape is [784, 128]
  // During forward: input [batch, 784] @ weights [784, 128] = output [batch,
  // 128]
  Matrix* weight_data = matrix_create(input_size, output_size);
  layer->weights = tensor_create(weight_data, 1);  // requires_grad = 1

  // Create bias tensor: [1, output_size] with gradient tracking
  // Bias is broadcasted across batch dimension during forward pass
  // Example: For fc1, bias shape is [1, 128], added to each sample in batch
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

// ============================================================================
// FORWARD PASS: output = input @ weights + bias
// ============================================================================
// Now fully using autograd for both weights and bias gradients
//
// Shape transformations:
//   input:      [batch_size, input_size]
//   weights:    [input_size, output_size]
//   matmul_out: [batch_size, output_size] = input @ weights
//   bias:       [1, output_size]
//   output:     [batch_size, output_size] = matmul_out + bias (broadcast)
//
// Example (fc1: 784 → 128 with batch_size=32):
//   input:      [32, 784]
//   weights:    [784, 128]
//   matmul_out: [32, 128]
//   bias:       [1, 128] → broadcasted to each of 32 samples
//   output:     [32, 128]
Tensor* linear_forward(Linear* layer, Tensor* input) {
  // Matrix multiplication: input @ weights
  // Shape: [batch, in] @ [in, out] = [batch, out]
  // Autograd builds computational graph for weight gradients
  Tensor* matmul_out = tensor_matmul(input, layer->weights);

  // Bias addition with broadcasting: [batch, out] + [1, out] = [batch, out]
  // Autograd's broadcast_add operation automatically:
  //   - Forward: adds bias[j] to matmul_out[i][j] for all i (batch samples)
  //   - Backward: sums gradients across batch dimension for bias gradient
  Tensor* output = tensor_broadcast_add(matmul_out, layer->bias);

  // Everything tracked through computational graph!
  // During backward pass:
  //   - weight gradients: computed via matmul backward rule
  //   - bias gradients: computed via broadcast_add backward rule (sum across
  //   batch)
  return output;
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
Matrix* matrix_from_tensor(Tensor* t) { return matrix_copy(t->data); }