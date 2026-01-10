#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "nn.h"
#include "optimizer.h"

#define ASSERT(condition, message)            \
  if (!(condition)) {                         \
    fprintf(stderr, "FAILED: %s\n", message); \
    return 0;                                 \
  } else {                                    \
    printf("PASSED: %s\n", message);          \
  }

int test_matrix_operations() {
  printf("\n=== Testing Matrix Operations ===\n");

  // Test matrix creation
  Matrix* m1 = matrix_create(2, 3);
  ASSERT(m1 != NULL && m1->rows == 2 && m1->cols == 3, "Matrix creation");

  // Test matrix fill
  matrix_fill(m1, 2.0f);
  ASSERT(m1->data[0] == 2.0f && m1->data[5] == 2.0f, "Matrix fill");

  // Test matrix addition
  Matrix* m2 = matrix_create(2, 3);
  matrix_fill(m2, 3.0f);
  Matrix* m3 = matrix_add(m1, m2);
  ASSERT(m3->data[0] == 5.0f, "Matrix addition");

  // Test matrix multiplication (element-wise)
  Matrix* m4 = matrix_mul(m1, m2);
  ASSERT(m4->data[0] == 6.0f, "Element-wise multiplication");

  // Test matrix transpose
  Matrix* m5 = matrix_create(2, 3);
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      m5->data[i * 3 + j] = i * 3 + j;
    }
  }
  Matrix* m6 = matrix_transpose(m5);
  ASSERT(m6->rows == 3 && m6->cols == 2, "Transpose dimensions");
  ASSERT(m6->data[0] == 0.0f && m6->data[1] == 3.0f, "Transpose values");

  // Test matrix multiplication (matmul)
  Matrix* a = matrix_create(2, 3);
  Matrix* b = matrix_create(3, 2);
  matrix_ones(a);
  matrix_ones(b);
  Matrix* c = matrix_matmul(a, b);
  ASSERT(c->rows == 2 && c->cols == 2, "Matmul dimensions");
  ASSERT(fabsf(c->data[0] - 3.0f) < 1e-5f, "Matmul values");

  // Cleanup
  matrix_free(m1);
  matrix_free(m2);
  matrix_free(m3);
  matrix_free(m4);
  matrix_free(m5);
  matrix_free(m6);
  matrix_free(a);
  matrix_free(b);
  matrix_free(c);

  return 1;
}

int test_activation_functions() {
  printf("\n=== Testing Activation Functions ===\n");

  // Test ReLU
  Matrix* m1 = matrix_create(1, 4);
  m1->data[0] = -2.0f;
  m1->data[1] = -1.0f;
  m1->data[2] = 1.0f;
  m1->data[3] = 2.0f;

  Matrix* relu_out = matrix_relu(m1);
  ASSERT(relu_out->data[0] == 0.0f && relu_out->data[2] == 1.0f,
         "ReLU activation");

  // Test Sigmoid
  Matrix* m2 = matrix_create(1, 2);
  m2->data[0] = 0.0f;
  m2->data[1] = 100.0f;

  Matrix* sig_out = matrix_sigmoid(m2);
  ASSERT(fabsf(sig_out->data[0] - 0.5f) < 1e-5f, "Sigmoid at 0");
  ASSERT(sig_out->data[1] > 0.99f, "Sigmoid at large positive");

  // Test Softmax
  Matrix* m3 = matrix_create(2, 3);
  for (size_t i = 0; i < 6; i++) {
    m3->data[i] = i;
  }
  Matrix* soft_out = matrix_softmax(m3);

  // Check that rows sum to 1
  float row1_sum = soft_out->data[0] + soft_out->data[1] + soft_out->data[2];
  ASSERT(fabsf(row1_sum - 1.0f) < 1e-5f, "Softmax row sum");

  // Cleanup
  matrix_free(m1);
  matrix_free(m2);
  matrix_free(m3);
  matrix_free(relu_out);
  matrix_free(sig_out);
  matrix_free(soft_out);

  return 1;
}

int test_linear_layer() {
  printf("\n=== Testing Linear Layer ===\n");

  // Create layer
  Linear* layer = linear_create(4, 3);
  ASSERT(layer != NULL, "Linear layer creation");
  ASSERT(layer->weights->rows == 4 && layer->weights->cols == 3,
         "Linear layer dimensions");

  // Initialize weights
  linear_init_xavier(layer);
  ASSERT(layer->bias->data[0] == 0.0f, "Bias initialization");

  // Test forward pass
  Matrix* input = matrix_create(2, 4);  // Batch of 2
  matrix_ones(input);

  Matrix* output = linear_forward(layer, input);
  ASSERT(output->rows == 2 && output->cols == 3, "Linear forward dimensions");

  // Test backward pass
  Matrix* grad_output = matrix_create(2, 3);
  matrix_ones(grad_output);

  Matrix* grad_input = NULL;
  linear_backward(layer, grad_output, &grad_input);
  ASSERT(grad_input != NULL, "Linear backward");
  ASSERT(grad_input->rows == 2 && grad_input->cols == 4,
         "Linear backward dimensions");

  // Cleanup
  matrix_free(input);
  matrix_free(output);
  matrix_free(grad_output);
  matrix_free(grad_input);
  linear_free(layer);

  return 1;
}

int test_optimizer() {
  printf("\n=== Testing Optimizer ===\n");

  // Create optimizer
  SGD* optimizer = sgd_create(0.1f, 0.0f, 0.0f);
  ASSERT(optimizer != NULL, "SGD creation");
  ASSERT(optimizer->learning_rate == 0.1f, "SGD learning rate");

  // Create a simple layer
  Linear* layer = linear_create(2, 2);
  matrix_fill(layer->weights, 1.0f);
  matrix_fill(layer->weight_grad, 0.5f);

  // Store original weight
  float original_weight = layer->weights->data[0];

  // Perform optimization step
  Linear* layers[] = {layer};
  sgd_step(optimizer, layers, 1);

  // Check that weights were updated
  float expected_weight = original_weight - 0.1f * 0.5f;
  ASSERT(fabsf(layer->weights->data[0] - expected_weight) < 1e-5f,
         "SGD weight update");

  // Cleanup
  linear_free(layer);
  sgd_free(optimizer);

  return 1;
}

int test_simple_training() {
  printf("\n=== Testing Simple Training Loop ===\n");

  // Create a simple model: 2 -> 4 -> 2
  Linear* layer1 = linear_create(2, 4);
  Linear* layer2 = linear_create(4, 2);
  linear_init_he(layer1);
  linear_init_xavier(layer2);

  // Create optimizer
  SGD* optimizer = sgd_create(0.01f, 0.0f, 0.0f);

  // Create simple XOR-like problem
  Matrix* x = matrix_create(4, 2);
  x->data[0] = 0.0f;
  x->data[1] = 0.0f;  // [0, 0]
  x->data[2] = 0.0f;
  x->data[3] = 1.0f;  // [0, 1]
  x->data[4] = 1.0f;
  x->data[5] = 0.0f;  // [1, 0]
  x->data[6] = 1.0f;
  x->data[7] = 1.0f;  // [1, 1]

  Matrix* y = matrix_create(4, 2);
  y->data[0] = 1.0f;
  y->data[1] = 0.0f;  // Class 0
  y->data[2] = 0.0f;
  y->data[3] = 1.0f;  // Class 1
  y->data[4] = 0.0f;
  y->data[5] = 1.0f;  // Class 1
  y->data[6] = 1.0f;
  y->data[7] = 0.0f;  // Class 0

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  // Train for a few iterations
  for (int iter = 0; iter < 100; iter++) {
    // Zero gradients
    linear_zero_grad(layer1);
    linear_zero_grad(layer2);

    // Forward pass
    Matrix* h1 = linear_forward(layer1, x);
    Matrix* a1 = matrix_relu(h1);
    Matrix* h2 = linear_forward(layer2, a1);
    Matrix* pred = matrix_softmax(h2);

    // Compute loss
    float loss = matrix_cross_entropy(pred, y);
    if (iter == 0) initial_loss = loss;
    if (iter == 99) final_loss = loss;

    // Backward pass
    Matrix* grad = matrix_sub(pred, y);

    Matrix* grad_a1 = NULL;
    linear_backward(layer2, grad, &grad_a1);

    Matrix* relu_mask = matrix_relu_derivative(h1);
    Matrix* grad_h1 = matrix_mul(grad_a1, relu_mask);

    linear_backward(layer1, grad_h1, NULL);

    // Update weights
    Linear* layers[] = {layer1, layer2};
    sgd_step(optimizer, layers, 2);

    // Cleanup
    matrix_free(h1);
    matrix_free(a1);
    matrix_free(h2);
    matrix_free(pred);
    matrix_free(grad);
    matrix_free(grad_a1);
    matrix_free(relu_mask);
    matrix_free(grad_h1);
  }

  printf("  Initial loss: %.4f\n", initial_loss);
  printf("  Final loss: %.4f\n", final_loss);
  ASSERT(final_loss < initial_loss, "Training reduces loss");

  // Cleanup
  matrix_free(x);
  matrix_free(y);
  linear_free(layer1);
  linear_free(layer2);
  sgd_free(optimizer);

  return 1;
}

int main() {
  printf("===========================================\n");
  printf("  C ML Library Test Suite\n");
  printf("===========================================\n");

  int all_passed = 1;

  all_passed &= test_matrix_operations();
  all_passed &= test_activation_functions();
  all_passed &= test_linear_layer();
  all_passed &= test_optimizer();
  all_passed &= test_simple_training();

  printf("\n===========================================\n");
  if (all_passed) {
    printf("  ✓ ALL TESTS PASSED\n");
  } else {
    printf("  ✗ SOME TESTS FAILED\n");
  }
  printf("===========================================\n");

  return all_passed ? 0 : 1;
}
