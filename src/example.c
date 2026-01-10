#include <stdio.h>

#include "matrix.h"
#include "nn.h"
#include "optimizer.h"

/*
 * Example: Building a simple 2-layer neural network from scratch
 * This demonstrates the core API of the ML library
 */

int main() {
  printf("=== C ML Library Usage Examples ===\n\n");

  // ========================================
  // 1. Matrix Operations
  // ========================================
  printf("1. Matrix Operations\n");
  printf("--------------------\n");

  // Create matrices
  Matrix* A = matrix_create(2, 3);
  Matrix* B = matrix_create(2, 3);

  // Fill with values
  printf("Matrix A:\n");
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      matrix_set(A, i, j, i * 3 + j);
    }
  }
  matrix_print(A);

  printf("\nMatrix B (filled with 2.0):\n");
  matrix_fill(B, 2.0f);
  matrix_print(B);

  // Matrix addition
  Matrix* C = matrix_add(A, B);
  printf("\nA + B:\n");
  matrix_print(C);

  // Matrix multiplication (element-wise)
  Matrix* D = matrix_mul(A, B);
  printf("\nA * B (element-wise):\n");
  matrix_print(D);

  // Matrix transpose
  Matrix* A_t = matrix_transpose(A);
  printf("\nA^T:\n");
  matrix_print(A_t);

  // Matrix matmul
  Matrix* E = matrix_matmul(A, A_t);
  printf("\nA @ A^T:\n");
  matrix_print(E);

  // Cleanup
  matrix_free(A);
  matrix_free(B);
  matrix_free(C);
  matrix_free(D);
  matrix_free(A_t);
  matrix_free(E);

  // ========================================
  // 2. Activation Functions
  // ========================================
  printf("\n2. Activation Functions\n");
  printf("------------------------\n");

  Matrix* X = matrix_create(1, 5);
  X->data[0] = -2.0f;
  X->data[1] = -1.0f;
  X->data[2] = 0.0f;
  X->data[3] = 1.0f;
  X->data[4] = 2.0f;

  printf("Input:\n");
  matrix_print(X);

  Matrix* relu_out = matrix_relu(X);
  printf("\nReLU(X):\n");
  matrix_print(relu_out);

  Matrix* sigmoid_out = matrix_sigmoid(X);
  printf("\nSigmoid(X):\n");
  matrix_print(sigmoid_out);

  // Softmax (create 2D matrix for proper softmax)
  Matrix* logits = matrix_create(2, 3);
  logits->data[0] = 1.0f;
  logits->data[1] = 2.0f;
  logits->data[2] = 3.0f;
  logits->data[3] = 1.0f;
  logits->data[4] = 2.0f;
  logits->data[5] = 3.0f;

  printf("\nLogits:\n");
  matrix_print(logits);

  Matrix* softmax_out = matrix_softmax(logits);
  printf("\nSoftmax(Logits):\n");
  matrix_print(softmax_out);

  // Cleanup
  matrix_free(X);
  matrix_free(relu_out);
  matrix_free(sigmoid_out);
  matrix_free(logits);
  matrix_free(softmax_out);

  // ========================================
  // 3. Neural Network Layer
  // ========================================
  printf("\n3. Linear Layer (Fully Connected)\n");
  printf("----------------------------------\n");

  // Create a linear layer: 3 inputs -> 2 outputs
  Linear* layer = linear_create(3, 2);
  linear_init_xavier(layer);

  printf("Layer: 3 -> 2\n");
  printf("Weights (3x2):\n");
  matrix_print(layer->weights);
  printf("\nBias (1x2):\n");
  matrix_print(layer->bias);

  // Forward pass with batch of 2 samples
  Matrix* input = matrix_create(2, 3);
  matrix_random(input, 0.0f, 1.0f);

  printf("\nInput batch (2x3):\n");
  matrix_print(input);

  Matrix* output = linear_forward(layer, input);
  printf("\nOutput (2x2):\n");
  matrix_print(output);

  // Cleanup
  matrix_free(input);
  matrix_free(output);
  linear_free(layer);

  // ========================================
  // 4. Simple Training Example
  // ========================================
  printf("\n4. Simple Training Example\n");
  printf("--------------------------\n");
  printf("Training a 2-layer network on dummy data\n\n");

  // Create model: 4 -> 8 -> 3
  Linear* fc1 = linear_create(4, 8);
  Linear* fc2 = linear_create(8, 3);
  linear_init_he(fc1);
  linear_init_xavier(fc2);

  // Create optimizer
  SGD* optimizer = sgd_create(0.05f, 0.0f, 0.0f);

  // Create dummy training data (batch_size=4)
  Matrix* train_x = matrix_create(4, 4);
  matrix_random(train_x, 0.0f, 1.0f);

  Matrix* train_y = matrix_create(4, 3);
  matrix_zeros(train_y);
  // One-hot labels
  matrix_set(train_y, 0, 0, 1.0f);
  matrix_set(train_y, 1, 1, 1.0f);
  matrix_set(train_y, 2, 2, 1.0f);
  matrix_set(train_y, 3, 0, 1.0f);

  // Training loop
  for (int epoch = 0; epoch < 50; epoch++) {
    // Zero gradients
    linear_zero_grad(fc1);
    linear_zero_grad(fc2);

    // Forward pass
    Matrix* h1 = linear_forward(fc1, train_x);
    Matrix* a1 = matrix_relu(h1);
    Matrix* h2 = linear_forward(fc2, a1);
    Matrix* pred = matrix_softmax(h2);

    // Compute loss
    float loss = matrix_cross_entropy(pred, train_y);

    // Backward pass
    Matrix* grad = matrix_sub(pred, train_y);

    Matrix* grad_a1 = NULL;
    linear_backward(fc2, grad, &grad_a1);

    Matrix* relu_mask = matrix_relu_derivative(h1);
    Matrix* grad_h1 = matrix_mul(grad_a1, relu_mask);

    linear_backward(fc1, grad_h1, NULL);

    // Update weights
    Linear* layers[] = {fc1, fc2};
    sgd_step(optimizer, layers, 2);

    // Print progress every 10 epochs
    if ((epoch + 1) % 10 == 0) {
      printf("Epoch %2d: Loss = %.4f\n", epoch + 1, loss);
    }

    // Cleanup iteration
    matrix_free(h1);
    matrix_free(a1);
    matrix_free(h2);
    matrix_free(pred);
    matrix_free(grad);
    matrix_free(grad_a1);
    matrix_free(relu_mask);
    matrix_free(grad_h1);
  }

  // Final prediction
  printf("\nFinal predictions:\n");
  Matrix* h1_final = linear_forward(fc1, train_x);
  Matrix* a1_final = matrix_relu(h1_final);
  Matrix* h2_final = linear_forward(fc2, a1_final);
  Matrix* pred_final = matrix_softmax(h2_final);
  matrix_print(pred_final);

  printf("\nTrue labels:\n");
  matrix_print(train_y);

  // Cleanup
  matrix_free(train_x);
  matrix_free(train_y);
  matrix_free(h1_final);
  matrix_free(a1_final);
  matrix_free(h2_final);
  matrix_free(pred_final);
  linear_free(fc1);
  linear_free(fc2);
  sgd_free(optimizer);

  printf("\n=== Examples Complete ===\n");

  return 0;
}
